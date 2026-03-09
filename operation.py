"""
Utility operations used by the local scattering-network implementation.

This file is adapted from a workflow built on top of the open-source
`scatseisnet` project:
https://github.com/scatseisnet/scatseisnet

Recommended upstream citation:
    Seydoux, L. S., Steinmann, R., Gärtner, M., Tong, F., Esfahani, R.,
    & Campillo, M. (2025). Scatseisnet, a Scattering network for seismic
    data analysis (0.3). Zenodo. https://doi.org/10.5281/zenodo.15110686
"""

from __future__ import annotations

import typing as T

import torch


def segment(
    x: torch.Tensor,
    window_size: int,
    stride: T.Optional[int] = None,
) -> T.Generator[torch.Tensor, None, None]:
    """Yield sliding windows from the last dimension of a tensor."""
    bins = x.shape[-1]
    index = 0
    stride = window_size if stride is None else stride
    while (index + window_size) <= bins:
        yield x[..., index : index + window_size]
        index += stride


def segmentize(
    x: torch.Tensor,
    window_size: int,
    stride: T.Optional[int] = None,
) -> torch.Tensor:
    """Stack the generator returned by :func:`segment` into a tensor."""
    return torch.stack([chunk for chunk in segment(x, window_size, stride)])


def pool(
    x: torch.Tensor,
    reduce_type: T.Optional[T.Callable] = None,
    pooling_factor: int = 1,
) -> torch.Tensor:
    """
    Pool the last tensor dimension with a reduction operator.

    Parameters
    ----------
    x
        Input tensor.
    reduce_type
        Reduction callable such as ``torch.max`` or ``torch.mean``-like
        functions returning a tuple or tensor.
    pooling_factor
        Number of adjacent samples grouped in the last dimension.
    """
    if reduce_type is None:
        return x

    new_shape = list(x.shape)
    new_shape[-1] = new_shape[-1] // pooling_factor
    reshaped = x.reshape(*new_shape[:-1], -1, pooling_factor)
    pooled = reduce_type(reshaped, dim=-1)
    return pooled[0] if isinstance(pooled, tuple) else pooled
