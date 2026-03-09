"""
Local scattering-network implementation used by the DeepCluster pipeline.

This module is adapted from and inspired by the open-source
`scatseisnet` project:
https://github.com/scatseisnet/scatseisnet

Recommended upstream citation:
    Seydoux, L. S., Steinmann, R., Gärtner, M., Tong, F., Esfahani, R.,
    & Campillo, M. (2025). Scatseisnet, a Scattering network for seismic
    data analysis (0.3). Zenodo. https://doi.org/10.5281/zenodo.15110686

This local version keeps the core Morlet-bank / scattering workflow while
serving as a lightweight, self-contained dependency for the downstream
clustering and representation-learning pipeline.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn

from operation import pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_window(
    x: torch.Tensor,
    width: T.Union[float, T.Sequence[float], torch.Tensor],
) -> torch.Tensor:
    """Compute a Gaussian window."""
    width = width if isinstance(width, torch.Tensor) else torch.tensor(width, requires_grad=True, device=DEVICE)
    width = width.unsqueeze(-1) if width.ndim == 1 else width
    return torch.exp(-((x / width.to(x.device)) ** 2))


def complex_morlet(
    x: torch.Tensor,
    center: T.Union[float, T.Sequence[float], torch.Tensor],
    width: T.Union[float, T.Sequence[float], torch.Tensor],
) -> torch.Tensor:
    """Construct a complex Morlet wavelet bank."""
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x, requires_grad=True, device=DEVICE)
    width = width if isinstance(width, torch.Tensor) else torch.tensor(width, requires_grad=True, device=DEVICE)
    center = center if isinstance(center, torch.Tensor) else torch.tensor(center, requires_grad=True, device=DEVICE)

    width = width.unsqueeze(-1) if width.ndim == 1 else width
    center = center.unsqueeze(-1) if center.ndim == 1 else center
    if width.shape != center.shape:
        raise ValueError(f"Widths shape {width.shape} and centers shape {center.shape} must match.")

    return gaussian_window(x, width.to(x.device)) * torch.exp(2j * torch.pi * center.to(x.device) * x)


class ComplexMorletBank(nn.Module):
    """Complex Morlet filter bank."""

    def __init__(
        self,
        bins: int,
        octaves: int = 8,
        resolution: int = 1,
        quality: T.Union[float, T.Sequence[float], torch.Tensor] = 4.0,
        centers: T.Optional[T.Union[float, T.Sequence[float], torch.Tensor]] = None,
        taper_alpha: T.Optional[float] = None,
        sampling_rate: float = 1.0,
        widths: T.Optional[T.Union[float, T.Sequence[float], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.octaves = octaves
        self.resolution = resolution
        self.sampling_rate = sampling_rate
        self.taper_alpha = taper_alpha

        self.quality = (
            nn.Parameter(torch.tensor(quality, dtype=torch.float32, device=DEVICE))
            if isinstance(quality, (float, list))
            else nn.Parameter(quality.to(DEVICE))
        )

        if centers is None:
            self._centers = nn.Parameter(self._calculate_centers().to(DEVICE))
        else:
            self._centers = (
                nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=DEVICE))
                if isinstance(centers, (float, list))
                else nn.Parameter(centers.to(DEVICE))
            )

        if widths is None:
            self._widths = self._calculate_widths()
        else:
            self._widths = widths if isinstance(widths, torch.Tensor) else torch.tensor(
                widths,
                requires_grad=True,
                device=DEVICE,
            )

        self.wavelets = complex_morlet(self.times, self._centers, self._widths)
        self.spectra = torch.fft.fft(self.wavelets)
        self.size = self.wavelets.shape[0]

    @property
    def times(self) -> torch.Tensor:
        """Time vector centered on zero."""
        duration = self.bins / self.sampling_rate
        return torch.linspace(-0.5, 0.5, steps=self.bins, device=DEVICE) * duration

    @property
    def frequencies(self) -> torch.Tensor:
        """Frequency vector."""
        return torch.linspace(0, self.sampling_rate, self.bins, device=DEVICE)

    @property
    def nyquist(self) -> float:
        """Nyquist frequency."""
        return self.sampling_rate / 2

    @property
    def shape(self) -> tuple[int, int]:
        """Wavelet bank shape."""
        return len(self), self.bins

    @property
    def ratios(self) -> torch.Tensor:
        """Wavelet bank logarithmic ratios."""
        octaves_value = self.octaves.item() if isinstance(self.octaves, torch.Tensor) else self.octaves
        ratios = torch.linspace(octaves_value, 0.0, self.shape[0] + 1, device=DEVICE)[:-1]
        return -ratios.flip(0)

    @property
    def scales(self) -> torch.Tensor:
        """Wavelet scales."""
        return 2 ** self.ratios

    @property
    def centers(self) -> torch.Tensor:
        """Wavelet center frequencies."""
        return self._centers

    @property
    def widths(self) -> torch.Tensor:
        """Wavelet widths."""
        return self._widths

    def _calculate_centers(self) -> torch.Tensor:
        return self.scales * self.nyquist

    def _calculate_widths(self) -> torch.Tensor:
        self.quality = self.quality.to(DEVICE)
        self._centers = self._centers.to(DEVICE)
        return self.quality / self._centers

    def __len__(self) -> int:
        return int(self.octaves * self.resolution)

    def __repr__(self) -> str:
        return (
            f"ComplexMorletBank(bins={self.bins}, octaves={self.octaves}, "
            f"resolution={self.resolution}, quality={self.quality}, "
            f"centers={self.centers}, sampling_rate={self.sampling_rate}, len={len(self)})"
        )

    def forward(self, segment: torch.Tensor) -> torch.Tensor:
        """Apply the wavelet bank to a segment."""
        device = segment.device
        spectra = self.spectra.to(device).clone()

        if self.taper_alpha is None:
            taper = torch.ones(segment.shape[-1], device=device)
        else:
            taper = torch.hann_window(segment.shape[-1], device=device)

        segment = torch.fft.fft(segment * taper)

        if segment.shape[-1] != spectra.shape[-1]:
            if segment.shape[-1] < spectra.shape[-1]:
                spectra = spectra[..., : segment.shape[-1]]
            else:
                padding = segment.shape[-1] - spectra.shape[-1]
                spectra = torch.nn.functional.pad(spectra, (0, padding))

        convolved = segment.unsqueeze(-2) * spectra
        scalogram = torch.fft.fftshift(torch.fft.ifft(convolved), dim=-1)
        return torch.abs(scalogram)


class ScatteringNetwork(nn.Module):
    """Multi-layer scattering network based on Morlet filter banks."""

    def __init__(
        self,
        *layer_kwargs: dict,
        bins: int = 128,
        sampling_rate: float = 1.0,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.bins = bins
        self.downsample_factor = layer_kwargs[0].get("downsample_factor", 4)

        cleaned_kwargs = []
        for kw in layer_kwargs:
            kw = kw.copy()
            kw.pop("downsample_factor", None)
            cleaned_kwargs.append(kw)

        self.banks = nn.ModuleList(
            [ComplexMorletBank(bins, sampling_rate=sampling_rate, **kw) for kw in cleaned_kwargs]
        )

    def transform_segment(
        self,
        segment: torch.Tensor,
        reduce_type: T.Optional[T.Callable] = None,
        pooling_factor: int = 1,
    ) -> list[torch.Tensor]:
        """Transform one segment through all scattering layers."""
        output = []
        for bank in self.banks:
            scalogram = bank(segment)
            scalogram = scalogram[..., :: self.downsample_factor]
            segment = scalogram
            output.append(pool(scalogram, reduce_type, pooling_factor))
        return output

    def forward(
        self,
        segments: torch.Tensor,
        reduce_type: T.Optional[T.Callable] = None,
        pooling_factor: int = 1,
    ) -> list[torch.Tensor]:
        """Transform a batch of segments."""
        features = [[] for _ in range(len(self.banks))]
        for segment in segments:
            scatterings = self.transform_segment(segment, reduce_type, pooling_factor)
            for layer_index, scattering in enumerate(scatterings):
                features[layer_index].append(scattering)
        return [torch.stack(feature) for feature in features]

    def __len__(self) -> int:
        return len(self.banks)

    def __repr__(self) -> str:
        header = (
            f"{self.__class__.__name__}(bins={self.bins}, "
            f"sampling_rate={self.sampling_rate}, len={len(self)})"
        )
        banks = "\n".join(str(bank) for bank in self.banks)
        return f"{header}\n{banks}"


def scattering_network(
    *layer_kwargs: dict,
    bins: int = 128,
    sampling_rate: float = 1.0,
) -> ScatteringNetwork:
    """Factory function for ScatteringNetwork."""
    return ScatteringNetwork(*layer_kwargs, bins=bins, sampling_rate=sampling_rate)
