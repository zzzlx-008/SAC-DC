#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepCluster with scattering features and attention pooling.

This script builds on and modifies the scattering-network implementation from
the open-source `scatseisnet` project:
https://github.com/scatseisnet/scatseisnet

Upstream repository:
    Seydoux, L. S., Steinmann, R., Gärtner, M., Tong, F., Esfahani, R.,
    & Campillo, M. (2025). Scatseisnet, a Scattering network for seismic
    data analysis (0.3). Zenodo. https://doi.org/10.5281/zenodo.15110686

If you redistribute this script or a repository containing it, please retain
this attribution and make sure your repository license remains compatible with
the upstream license terms.

Relative to the upstream scattering implementation, this project adds:
    1. waveform preprocessing and timestamp parsing for seismic segments,
    2. a DeepCluster-style iterative pseudo-label training loop,
    3. attention pooling for fixed-length embedding extraction,
    4. full-dataset KMeans clustering without PCA,
    5. representative-waveform visualization and export utilities.

The current script is intended for research use and large-scale seismic /
planetary-radar-inspired segment clustering workflows.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from obspy import UTCDateTime
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from scatnet import scattering_network

LOGGER = logging.getLogger("deepcluster_attn_kmeans")


@dataclass
class Config:
    """Experiment configuration."""

    # Data
    segment_duration_seconds: int = 120
    sampling_rate_hertz: float = 6.625
    samples_per_segment: int = 0

    # Model
    feature_dim: int = 128
    num_clusters: int = 40

    # Multi-scale temporal pooling for raw waveform features
    raw_pool_lengths: Tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8)

    # Training
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    subset_size: int = 10000
    clustering_interval: int = 3

    # KMeans
    kmeans_max_iter: int = 300
    kmeans_n_init: int = 10
    reset_head_each_cluster: bool = True

    # Attention pooling
    attn_tokens: int = 16
    attn_heads: int = 4
    attn_layers: int = 2
    attn_dropout: float = 0.1

    # Scattering network
    o1: int = 6
    r1: int = 2
    o2: int = 4
    r2: int = 2
    downsample_factor: int = 4

    # Misc
    random_seed: int = 42

    # Runtime I/O
    station: str = "S12"
    channel: str = "MH2"
    segments_file: str = ""
    timestamps_file: str = ""
    output_dir: str = ""


def setup_logging() -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_1d(x: np.ndarray) -> np.ndarray:
    """Standardize a 1D waveform."""
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd + 1e-8)


def parse_timestamp(ts: object) -> float:
    """
    Convert various timestamp formats to Unix time in UTC seconds.

    Supported input types:
        - obspy.UTCDateTime
        - numpy.datetime64
        - datetime.datetime
        - string timestamps
        - numeric values
    """
    if isinstance(ts, UTCDateTime):
        return float(ts.timestamp)

    if isinstance(ts, np.datetime64):
        return float(ts.astype("datetime64[s]").astype(np.int64))

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return float(ts.timestamp())

    if isinstance(ts, str):
        text = ts.strip()
        try:
            return float(UTCDateTime(text).timestamp)
        except Exception:
            pass

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
                return float(dt.timestamp())
            except ValueError:
                continue
        return 0.0

    try:
        return float(ts)
    except Exception:
        return 0.0


def is_uniform_array(arr: np.ndarray) -> bool:
    """Return True if all values in the array are identical."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return True
    return bool(np.all(arr == arr.flat[0]))


def pad_crop_to_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or crop a waveform to the requested length."""
    x = np.asarray(x)
    n_samples = len(x)
    if n_samples == target_length:
        return x
    if n_samples < target_length:
        return np.pad(x, (0, target_length - n_samples), mode="constant", constant_values=0.0)
    return x[:target_length]


def pick_num_workers() -> int:
    """Choose a moderate number of DataLoader workers."""
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


class IndexedSeismicDataset(Dataset):
    """Dataset returning waveform tensors and their global indices."""

    def __init__(self, data_tensor: torch.Tensor) -> None:
        self.data = data_tensor

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], idx


def replace_nan_with_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaNs in a tensor with column means."""
    col_means = torch.nanmean(tensor, dim=0, keepdim=True)
    col_means = torch.nan_to_num(col_means, nan=0.0)
    nan_mask = torch.isnan(tensor)
    out = tensor.clone()
    if nan_mask.any():
        out[nan_mask] = col_means[0, nan_mask[0]]
    return out


def pool_scattering_to_fixed_length(order: torch.Tensor, target_length: int = 32) -> torch.Tensor:
    """
    Pool scattering coefficients to a fixed temporal length.

    The function accepts 2D, 3D, 4D, or 5D tensors and returns shape (B, C, L).
    """
    original_shape = tuple(order.shape)
    original_numel = int(order.numel())

    if order.ndim == 5:
        batch = order.shape[0]
        length = order.shape[-1]
        middle = order.shape[1:-1]
        channels = int(np.prod([int(m) for m in middle]))
        if batch * channels * length != original_numel:
            length = order.shape[1]
            middle = order.shape[2:]
            channels = int(np.prod([int(m) for m in middle]))
        try:
            order = order.view(batch, channels, length)
        except RuntimeError:
            channels = original_numel // batch
            order = order.view(batch, channels, 1)

    elif order.ndim == 4:
        batch = order.shape[0]
        length = order.shape[-1]
        middle = order.shape[1:-1]
        channels = int(np.prod([int(m) for m in middle]))
        try:
            order = order.view(batch, channels, length)
        except RuntimeError:
            channels = original_numel // batch
            order = order.view(batch, channels, 1)

    elif order.ndim == 2:
        order = order.unsqueeze(1)

    if order.ndim != 3:
        raise ValueError(
            f"Expected pooled scattering tensor to be 3D after reshape, "
            f"but got {tuple(order.shape)} from original shape {original_shape}."
        )

    batch, channels, length = order.shape
    if length <= 0:
        return torch.zeros(batch, channels, target_length, device=order.device)

    kernel = max(1, length // target_length)
    if length % target_length != 0:
        kernel += 1

    pooled = F.avg_pool1d(order, kernel_size=kernel, stride=kernel)
    if pooled.shape[-1] < target_length:
        pooled = F.pad(pooled, (0, target_length - pooled.shape[-1]), mode="constant", value=0.0)

    return pooled[:, :, :target_length]


def adaptive_pool_to_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pool a (B, C, T) tensor to a requested temporal length."""
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D tensor (B, C, T), got shape {tuple(x.shape)}")
    return F.adaptive_avg_pool1d(x, output_size=target_length)


def build_multiscale_raw_features(
    original_wave: torch.Tensor,
    pool_lengths: Sequence[int],
    target_length: int,
) -> torch.Tensor:
    """
    Build multiscale raw-waveform features.

    Steps:
        1. adaptively average-pool the raw waveform to each requested temporal scale,
        2. upsample every pooled feature map back to ``target_length``,
        3. concatenate them along the channel axis.

    Input:
        original_wave: (B, 1, T)

    Output:
        (B, len(pool_lengths), target_length)
    """
    multiscale_features: List[torch.Tensor] = []
    for pooled_len in pool_lengths:
        pooled = adaptive_pool_to_length(original_wave, pooled_len)
        resized = F.interpolate(pooled, size=target_length, mode="linear", align_corners=False)
        multiscale_features.append(resized)
    return torch.cat(multiscale_features, dim=1)


def merge_features(
    raw_multiscale: torch.Tensor,
    order1_pooled: torch.Tensor,
    order2_pooled: torch.Tensor,
) -> torch.Tensor:
    """Concatenate multiscale raw-waveform and scattering features."""
    return torch.cat([raw_multiscale, order1_pooled, order2_pooled], dim=1)


class AttnBlock(nn.Module):
    """Transformer-style attention block for sequence pooling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ffn(x))


class AttnPool1D(nn.Module):
    """Attention-based pooling from (B, D, T) to (B, D)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, n_tokens: int, dropout: float) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_tokens, d_model))
        self.blocks = nn.ModuleList([AttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, D)
        batch, tokens, dim = x.shape

        if tokens != self.n_tokens:
            if tokens > self.n_tokens:
                x = x[:, : self.n_tokens, :]
            else:
                pad = self.n_tokens - tokens
                x = torch.cat([x, x.new_zeros(batch, pad, dim)], dim=1)

        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos

        for block in self.blocks:
            x = block(x)

        return x[:, 0, :]


class DeepClusterModel(nn.Module):
    """
    DeepCluster backbone with attention pooling.

    Input shape:
        (B, C, T_fused)

    Output:
        logits over cluster IDs
    """

    def __init__(self, in_channels: int, config: Config) -> None:
        super().__init__()
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(config.attn_tokens),
        )

        self.attn_pool = AttnPool1D(
            d_model=128,
            n_heads=config.attn_heads,
            n_layers=config.attn_layers,
            n_tokens=config.attn_tokens,
            dropout=config.attn_dropout,
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(128, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
        )

        self.head = nn.Linear(config.feature_dim, config.num_clusters)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_extractor(x)
        pooled = self.attn_pool(features)
        embedding = self.feature_projection(pooled)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def compute_input_channels(config: Config, device: torch.device, network: nn.Module) -> int:
    """Infer the number of fused feature channels."""
    dummy = torch.randn(1, config.samples_per_segment, device=device).unsqueeze(0)
    with torch.no_grad():
        order1, order2 = network(dummy)
    order1_pooled = pool_scattering_to_fixed_length(order1)
    order2_pooled = pool_scattering_to_fixed_length(order2)
    raw_channels = len(config.raw_pool_lengths)
    return int(raw_channels + order1_pooled.shape[1] + order2_pooled.shape[1])


def build_merged_from_wave(
    batch_wave: torch.Tensor,
    device: torch.device,
    network: nn.Module,
    original_downsampler: Optional[nn.Module],
    config: Config,
) -> torch.Tensor:
    """Build fused multiscale-raw and scattering features."""
    del original_downsampler  # kept in the function signature for backward compatibility

    batch_wave = batch_wave.to(device, non_blocking=True)
    original = batch_wave.unsqueeze(1)

    scattering_input = batch_wave.view(batch_wave.shape[0], 1, -1)
    order1, order2 = network(scattering_input)
    order1_pooled = pool_scattering_to_fixed_length(order1)
    order2_pooled = pool_scattering_to_fixed_length(order2)

    scattering_target_length = order1_pooled.shape[-1]
    raw_multiscale = build_multiscale_raw_features(
        original_wave=original,
        pool_lengths=config.raw_pool_lengths,
        target_length=scattering_target_length,
    )

    merged = merge_features(
        raw_multiscale=raw_multiscale,
        order1_pooled=order1_pooled,
        order2_pooled=order2_pooled,
    )
    return replace_nan_with_mean(merged)


@torch.inference_mode()
def extract_embeddings(
    model: DeepClusterModel,
    data_loader: DataLoader,
    device: torch.device,
    network: nn.Module,
    original_downsampler: nn.Module,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and corresponding global indices."""
    model.eval()
    network.eval()

    embeddings: List[torch.Tensor] = []
    global_indices: List[torch.Tensor] = []

    for wave, gidx in tqdm(data_loader, desc="Extract embeddings"):
        merged = build_merged_from_wave(wave, device, network, original_downsampler, config)
        feat = model.forward_features(merged)
        embeddings.append(feat.cpu())
        global_indices.append(gidx)

    emb = torch.cat(embeddings, dim=0).numpy().astype(np.float32, copy=False)
    idxs = torch.cat(global_indices, dim=0).numpy().astype(np.int64, copy=False)
    return emb, idxs


def kmeans_fit(config: Config, embeddings: np.ndarray, seed: int) -> Tuple[KMeans, np.ndarray]:
    """Run full KMeans on the given embedding matrix."""
    x = np.ascontiguousarray(embeddings, dtype=np.float32)
    kmeans = KMeans(
        n_clusters=config.num_clusters,
        random_state=seed,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
        verbose=0,
    )
    labels = kmeans.fit_predict(x).astype(np.int64, copy=False)
    return kmeans, labels


def train_deepcluster_supervised(
    model: DeepClusterModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    network: nn.Module,
    original_downsampler: nn.Module,
    config: Config,
    epoch: int,
    pseudo_labels_full: torch.Tensor,
) -> float:
    """Train one epoch using pseudo-label cross-entropy."""
    model.train()
    network.eval()

    total_loss = 0.0
    n_batches = 0

    for wave, gidx in tqdm(train_loader, desc=f"Train CE epoch {epoch}"):
        targets_cpu = pseudo_labels_full[gidx]
        valid_cpu = targets_cpu >= 0
        if valid_cpu.sum().item() == 0:
            continue

        wave = wave[valid_cpu]
        targets = targets_cpu[valid_cpu].to(device, non_blocking=True)

        with torch.no_grad():
            merged = build_merged_from_wave(wave, device, network, original_downsampler, config)

        optimizer.zero_grad(set_to_none=True)
        logits = model(merged)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    average_loss = total_loss / max(1, n_batches)
    LOGGER.info("Epoch %d CE loss: %.4f", epoch, average_loss)
    return average_loss


def visualize_centroid_waveforms(
    waves_tensor: torch.Tensor,
    cluster_labels: np.ndarray,
    timestamps_raw: Sequence[object],
    emb_full: np.ndarray,
    kmeans: KMeans,
    output_dir: str,
    title_prefix: str = "",
) -> Tuple[List[int], List[int]]:
    """Select and plot representative waveforms nearest to cluster centroids."""
    labels = cluster_labels.astype(np.int64, copy=False)
    n_clusters = int(kmeans.n_clusters)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    counts = np.bincount(labels, minlength=n_clusters)

    centroid_indices: List[int] = []
    for cluster_id in range(n_clusters):
        idxs = np.where(labels == cluster_id)[0]
        if idxs.size == 0:
            centroid_indices.append(-1)
            continue
        diff = emb_full[idxs] - centers[cluster_id][None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        centroid_indices.append(int(idxs[np.argmin(dist)]))

    valid_clusters = [cid for cid in range(n_clusters) if counts[cid] > 0]
    if len(valid_clusters) >= 2:
        valid_centers = centers[valid_clusters]
        link = linkage(valid_centers, method="ward")
        order = dendrogram(link, no_plot=True)["leaves"]
        sorted_cluster_ids = [valid_clusters[i] for i in order]
    else:
        sorted_cluster_ids = valid_clusters

    plt.figure(figsize=(20, 20))
    num_cols = 8
    num_rows = (len(sorted_cluster_ids) + num_cols - 1) // num_cols

    for panel_index, cluster_id in enumerate(sorted_cluster_ids):
        plt.subplot(num_rows, num_cols, panel_index + 1)
        idx = centroid_indices[cluster_id]
        if idx < 0:
            plt.text(0.5, 0.5, f"Cluster {cluster_id}\nNo data", ha="center", va="center", fontsize=9)
            plt.axis("off")
            continue

        waveform = waves_tensor[idx].numpy()
        plt.plot(waveform, linewidth=1.2)

        ts = timestamps_raw[idx]
        try:
            ts_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        except Exception:
            ts_str = str(ts)[:10]

        plt.title(f"{title_prefix}C{cluster_id}\n({counts[cluster_id]} samples)\n{ts_str}", fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.ylim(-3, 3)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    out_png = Path(output_dir) / "cluster_centroid_waveforms.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    LOGGER.info("Centroid waveforms saved: %s", out_png)

    return centroid_indices, sorted_cluster_ids


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="DeepCluster with multiscale raw-waveform and scattering-feature fusion (no PCA)."
    )
    parser.add_argument("--station", type=str, default="S12")
    parser.add_argument("--channel", type=str, default="MH2")
    parser.add_argument("--segments_file", type=str, default="")
    parser.add_argument("--timestamps_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")

    parser.add_argument("--num_clusters", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--clustering_interval", type=int, default=3)

    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--kmeans_n_init", type=int, default=10)

    parser.add_argument("--attn_tokens", type=int, default=16)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    return parser


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Populate a Config object from command-line arguments."""
    config = Config()
    config.station = args.station
    config.channel = args.channel
    config.num_clusters = args.num_clusters
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.subset_size = args.subset_size
    config.clustering_interval = args.clustering_interval
    config.kmeans_max_iter = args.kmeans_max_iter
    config.kmeans_n_init = args.kmeans_n_init
    config.attn_tokens = args.attn_tokens
    config.attn_heads = args.attn_heads
    config.attn_layers = args.attn_layers
    config.attn_dropout = args.attn_dropout
    config.samples_per_segment = int(config.segment_duration_seconds * config.sampling_rate_hertz)

    default_segments = os.path.join("output14", f"{config.station}_{config.channel}.npy")
    default_timestamps = os.path.join("output14", f"{config.station}_common_timestamps_datetime64_utc.npy")

    config.segments_file = args.segments_file or default_segments
    config.timestamps_file = args.timestamps_file or default_timestamps
    config.output_dir = args.output_dir or f"seismic_result_{config.station}_{config.channel}_attn_kmeans"
    return config


def load_and_preprocess_segments(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw arrays, filter invalid entries, and standardize segments."""
    if not os.path.exists(config.segments_file):
        raise FileNotFoundError(f"segments_file not found: {config.segments_file}")
    if not os.path.exists(config.timestamps_file):
        raise FileNotFoundError(f"timestamps_file not found: {config.timestamps_file}")

    LOGGER.info("Loading segments from %s", config.segments_file)
    LOGGER.info("Loading timestamps from %s", config.timestamps_file)

    segments = np.load(config.segments_file, allow_pickle=True)
    timestamps = np.load(config.timestamps_file, allow_pickle=True)

    if len(segments) != len(timestamps):
        n_common = min(len(segments), len(timestamps))
        LOGGER.warning(
            "segments/timestamps size mismatch: %d vs %d; truncating to %d",
            len(segments),
            len(timestamps),
            n_common,
        )
        segments = segments[:n_common]
        timestamps = timestamps[:n_common]

    valid_segments: List[np.ndarray] = []
    valid_timestamps: List[object] = []

    LOGGER.info("Filtering uniform or invalid segments.")
    for seg, ts in zip(segments, timestamps):
        seg = np.asarray(seg)
        if seg.size == 0 or is_uniform_array(seg):
            continue
        valid_segments.append(seg)
        valid_timestamps.append(ts)

    if not valid_segments:
        raise ValueError("No valid segments remain after filtering.")

    processed: List[np.ndarray] = []
    for seg in valid_segments:
        seg = pad_crop_to_length(np.asarray(seg, dtype=np.float32), config.samples_per_segment)
        processed.append(norm_1d(seg))

    processed_array = np.stack(processed, axis=0).astype(np.float32, copy=False)
    timestamp_objects = np.asarray(valid_timestamps, dtype=object)
    timestamp_values = np.asarray([parse_timestamp(ts) for ts in timestamp_objects], dtype=np.float64)

    return processed_array, timestamp_objects, timestamp_values


def save_run_artifacts(
    config: Config,
    processed: np.ndarray,
    waves_tensor: torch.Tensor,
    model: DeepClusterModel,
    optimizer: optim.Optimizer,
    loss_history: List[float],
    cluster_labels: np.ndarray,
    timestamp_values: np.ndarray,
    centroid_indices: List[int],
    sorted_cluster_ids: List[int],
) -> None:
    """Save model outputs and training artifacts."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "segments.npy", processed)
    torch.save(waves_tensor, output_dir / "filtered_waves.pt")

    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
        },
        output_dir / "deepcluster_model.pth",
    )

    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    np.save(output_dir / "timestamps.npy", timestamp_values)
    np.save(output_dir / "centroid_indices.npy", np.asarray(centroid_indices, dtype=np.int64))
    np.save(output_dir / "sorted_cluster_ids.npy", np.asarray(sorted_cluster_ids, dtype=np.int64))

    if loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("DeepCluster training loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "loss_history.png", dpi=150, bbox_inches="tight")
        plt.close()


def main() -> None:
    """Entry point."""
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    set_random_seed(config.random_seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Configuration: %s", asdict(config))

    processed, timestamps, timestamp_values = load_and_preprocess_segments(config)
    waves_tensor = torch.from_numpy(processed).to(torch.float32)
    dataset = IndexedSeismicDataset(waves_tensor)
    dataset_size = len(dataset)
    LOGGER.info("Final dataset size: %d", dataset_size)

    LOGGER.info("Initializing scattering network.")
    q1 = torch.randint(1, 3, (config.o1 * config.r1,), dtype=torch.float32)
    q2 = torch.randint(2, 4, (config.o2 * config.r2,), dtype=torch.float32)
    network = scattering_network(
        {
            "octaves": torch.tensor(config.o1),
            "resolution": torch.tensor(config.r1),
            "quality": q1,
            "downsample_factor": config.downsample_factor,
        },
        {
            "octaves": torch.tensor(config.o2),
            "resolution": torch.tensor(config.r2),
            "quality": q2,
        },
        bins=config.samples_per_segment,
        sampling_rate=config.sampling_rate_hertz,
    ).to(device)
    network.eval()

    original_downsampler = None

    in_channels = compute_input_channels(config, device, network)
    LOGGER.info("Raw multiscale pooling lengths: %s", list(config.raw_pool_lengths))
    LOGGER.info("Merged feature channels: %d", in_channels)
    model = DeepClusterModel(in_channels=in_channels, config=config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    LOGGER.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    subset_size = min(config.subset_size, dataset_size)
    subset_indices = torch.randperm(dataset_size)[:subset_size].tolist()
    subset = Subset(dataset, subset_indices)

    num_workers = pick_num_workers()
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    subset_loader_shuffle = DataLoader(subset, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    subset_loader_noshuf = DataLoader(subset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    full_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    LOGGER.info("Subset size: %d / %d", subset_size, dataset_size)
    LOGGER.info("DataLoader workers: %d", num_workers)

    pseudo_labels_full = torch.full((dataset_size,), -1, dtype=torch.long)
    loss_history: List[float] = []
    kmeans: Optional[KMeans] = None

    for epoch in range(1, config.num_epochs + 1):
        LOGGER.info("%s", "=" * 72)
        LOGGER.info("Epoch %d / %d", epoch, config.num_epochs)
        LOGGER.info("%s", "=" * 72)

        if (epoch - 1) % config.clustering_interval == 0:
            LOGGER.info("Re-clustering: extract embeddings and run KMeans.")
            emb, idxs = extract_embeddings(
                model=model,
                data_loader=subset_loader_noshuf,
                device=device,
                network=network,
                original_downsampler=original_downsampler,
                config=config,
            )
            kmeans, labels = kmeans_fit(config, emb, seed=config.random_seed + epoch)

            pseudo_labels_full.fill_(-1)
            pseudo_labels_full[torch.from_numpy(idxs).long()] = torch.from_numpy(labels).long()

            if config.reset_head_each_cluster:
                model.head.reset_parameters()
                LOGGER.info("Classifier head reset after re-clustering.")

            counts = np.bincount(labels, minlength=config.num_clusters)
            LOGGER.info("Non-empty clusters: %d / %d", int(np.sum(counts > 0)), config.num_clusters)
            LOGGER.info("Largest cluster sizes: %s", np.sort(counts)[-5:][::-1].tolist())

        epoch_loss = train_deepcluster_supervised(
            model=model,
            train_loader=subset_loader_shuffle,
            optimizer=optimizer,
            device=device,
            network=network,
            original_downsampler=original_downsampler,
            config=config,
            epoch=epoch,
            pseudo_labels_full=pseudo_labels_full,
        )
        loss_history.append(epoch_loss)

    if kmeans is None:
        raise RuntimeError("KMeans model was not initialized during training.")

    LOGGER.info("Final labeling on the full dataset.")
    full_emb, _ = extract_embeddings(
        model=model,
        data_loader=full_loader,
        device=device,
        network=network,
        original_downsampler=original_downsampler,
        config=config,
    )
    cluster_labels = kmeans.predict(np.ascontiguousarray(full_emb, dtype=np.float32)).astype(np.int64, copy=False)

    LOGGER.info("Visualizing representative centroid waveforms.")
    centroid_indices, sorted_cluster_ids = visualize_centroid_waveforms(
        waves_tensor=waves_tensor,
        cluster_labels=cluster_labels,
        timestamps_raw=timestamps,
        emb_full=full_emb,
        kmeans=kmeans,
        output_dir=config.output_dir,
        title_prefix=f"{config.station}_{config.channel}_",
    )

    save_run_artifacts(
        config=config,
        processed=processed,
        waves_tensor=waves_tensor,
        model=model,
        optimizer=optimizer,
        loss_history=loss_history,
        cluster_labels=cluster_labels,
        timestamp_values=timestamp_values,
        centroid_indices=centroid_indices,
        sorted_cluster_ids=sorted_cluster_ids,
    )

    LOGGER.info("Run completed successfully.")
    LOGGER.info("Output directory: %s", config.output_dir)


if __name__ == "__main__":
    main()
