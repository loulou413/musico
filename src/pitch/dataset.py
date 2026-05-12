"""PyTorch dataset for Saraga pitch annotations.

Frames audio at 16 kHz (CREPE's native rate), extracts 1024-sample windows,
and maps each window to a 360-bin pitch class label — the same bin space as
CREPE, so the trained model's outputs are directly comparable.

Frequency bins:  440 × 2^((i×20 − 6900) / 1200)  for i = 0..359
                 ≈ 32.7 Hz (C1) → 1975.5 Hz (B6), 20 cents per bin.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from src.preprocessing.common import AudioTrack

# ── CREPE-compatible frequency bins ──────────────────────────────────────────
N_BINS = 360
PITCH_BINS_HZ = 440.0 * 2.0 ** ((np.arange(N_BINS) * 20 - 6900) / 1200.0)
PITCH_BINS_CENTS = 1200.0 * np.log2(PITCH_BINS_HZ / 10.0)   # cents re 10 Hz

TARGET_SR = 16_000          # CREPE standard input rate
FRAME_LEN = 1024            # samples at 16 kHz  (~64 ms)
HOP_SAMPLES = 160           # 10 ms at 16 kHz


def hz_to_bin_index(freq_hz: float) -> int | None:
    """Return the nearest CREPE bin index for a frequency, or None if unvoiced."""
    if freq_hz <= 0:
        return None
    cents = 1200.0 * np.log2(freq_hz / 10.0)
    idx = int(np.argmin(np.abs(PITCH_BINS_CENTS - cents)))
    return idx


def freq_to_soft_label(freq_hz: float, sigma_bins: float = 1.0) -> np.ndarray:
    """Gaussian-smoothed one-hot over 360 bins (reduces overconfidence on gamakas)."""
    label = np.zeros(N_BINS, dtype=np.float32)
    if freq_hz <= 0:
        return label   # all-zeros = unvoiced
    cents = 1200.0 * np.log2(freq_hz / 10.0)
    dist = (PITCH_BINS_CENTS - cents) / (20.0 * sigma_bins)   # in bin widths
    label = np.exp(-0.5 * dist ** 2).astype(np.float32)
    s = label.sum()
    if s > 0:
        label /= s
    return label


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame - frame.mean()
    std = frame.std()
    return (frame / (std + 1e-8)).astype(np.float32)


class SaragaPitchDataset(Dataset):
    """Fixed-length audio frames paired with per-frame pitch bin labels.

    Args:
        tracks:        AudioTrack objects with f0_times / f0_freqs annotations.
        sigma_bins:    Gaussian width for label smoothing (0 = hard one-hot).
        max_gap_s:     Maximum time gap between a frame centre and its nearest
                       annotation before the frame is treated as unvoiced.
    """

    def __init__(
        self,
        tracks: list[AudioTrack],
        sigma_bins: float = 1.0,
        max_gap_s: float = 0.020,
    ):
        self.frames: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []     # (N_BINS,) soft label
        self.voiced: list[bool] = []

        for track in tracks:
            if track.f0_times is None or track.f0_freqs is None:
                continue
            self._process_track(track, sigma_bins, max_gap_s)

    def _process_track(self, track: AudioTrack, sigma_bins: float, max_gap_s: float):
        audio16 = _resample(track.audio, track.sr)
        n = len(audio16)
        half = FRAME_LEN // 2

        # Pad so every annotation time is centred in a frame
        audio_padded = np.pad(audio16, (half, half))

        for t_ref, f_ref in zip(track.f0_times, track.f0_freqs):
            centre = int(round(t_ref * TARGET_SR))
            start = centre          # after padding, centre + half is correct
            end = start + FRAME_LEN
            if end > len(audio_padded):
                break

            frame = audio_padded[start:end]
            frame = _normalize_frame(frame)

            label = freq_to_soft_label(f_ref, sigma_bins=sigma_bins)
            is_voiced = f_ref > 0

            self.frames.append(frame)
            self.labels.append(label)
            self.voiced.append(is_voiced)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame = torch.from_numpy(self.frames[idx]).unsqueeze(0)   # (1, FRAME_LEN)
        label = torch.from_numpy(self.labels[idx])                # (N_BINS,)
        voiced = torch.tensor(self.voiced[idx], dtype=torch.float32)
        return frame, label, voiced


def make_splits(
    tracks: list[AudioTrack],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    **dataset_kwargs,
) -> tuple["SaragaPitchDataset", "SaragaPitchDataset", "SaragaPitchDataset"]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(tracks)).tolist()
    n_train = int(len(tracks) * train_ratio)
    n_val   = int(len(tracks) * val_ratio)

    train_tracks = [tracks[i] for i in indices[:n_train]]
    val_tracks   = [tracks[i] for i in indices[n_train: n_train + n_val]]
    test_tracks  = [tracks[i] for i in indices[n_train + n_val:]]

    return (
        SaragaPitchDataset(train_tracks, **dataset_kwargs),
        SaragaPitchDataset(val_tracks,   **dataset_kwargs),
        SaragaPitchDataset(test_tracks,  **dataset_kwargs),
    )
