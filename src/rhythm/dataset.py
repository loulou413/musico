"""PyTorch dataset for Saraga beat annotations.

Each sample is a fixed-length mel spectrogram segment paired with a
per-frame binary beat label.  Tracks are split into non-overlapping
segments of `seq_len` frames so batching is straightforward.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, random_split

from src.preprocessing.common import AudioTrack


def compute_mel(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
) -> np.ndarray:
    """Return log-mel spectrogram (n_mels × T), normalised to [0, 1]."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    lo, hi = log_mel.min(), log_mel.max()
    if hi > lo:
        log_mel = (log_mel - lo) / (hi - lo)
    return log_mel.astype(np.float32)


def beats_to_frame_labels(
    beat_times: np.ndarray,
    n_frames: int,
    sr: int,
    hop_length: int,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """Binary beat label per mel frame.

    A narrow Gaussian blob around each beat frame avoids the hard 0/1
    boundary and gives the model a smoother gradient signal.
    """
    labels = np.zeros(n_frames, dtype=np.float32)
    frame_idx = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    frame_idx = frame_idx[(frame_idx >= 0) & (frame_idx < n_frames)]

    if gaussian_sigma <= 0:
        labels[frame_idx] = 1.0
        return labels

    half = int(np.ceil(2 * gaussian_sigma))
    t = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (t / gaussian_sigma) ** 2)
    for fi in frame_idx:
        lo = fi - half
        hi = fi + half + 1
        klo = max(0, -lo)
        khi = len(kernel) - max(0, hi - n_frames)
        labels[max(0, lo): min(n_frames, hi)] = np.maximum(
            labels[max(0, lo): min(n_frames, hi)],
            kernel[klo:khi],
        )
    return labels


class SaragaBeatDataset(Dataset):
    """Segments of (mel, beat_labels) drawn from a list of AudioTrack objects.

    Args:
        tracks:        AudioTrack objects with beat_times annotations.
        seq_len:       Number of mel frames per segment (default 512 ≈ 12 s).
        hop_length:    STFT hop in samples.
        n_mels:        Number of mel bands.
        gaussian_sigma: Width of Gaussian beat label (0 = hard binary).
    """

    def __init__(
        self,
        tracks: list[AudioTrack],
        seq_len: int = 512,
        hop_length: int = 512,
        n_mels: int = 128,
        gaussian_sigma: float = 1.0,
    ):
        self.seq_len = seq_len
        self.segments: list[tuple[np.ndarray, np.ndarray]] = []

        for track in tracks:
            if track.beat_times is None or len(track.beat_times) == 0:
                continue

            mel = compute_mel(track.audio, track.sr, hop_length=hop_length, n_mels=n_mels)
            n_frames = mel.shape[1]
            labels = beats_to_frame_labels(
                track.beat_times, n_frames, track.sr, hop_length, gaussian_sigma
            )

            for start in range(0, n_frames - seq_len + 1, seq_len):
                mel_seg = mel[:, start: start + seq_len]        # (n_mels, seq_len)
                lbl_seg = labels[start: start + seq_len]         # (seq_len,)
                self.segments.append((mel_seg, lbl_seg))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel, lbl = self.segments[idx]
        return torch.from_numpy(mel), torch.from_numpy(lbl)


def make_splits(
    tracks: list[AudioTrack],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    **dataset_kwargs,
) -> tuple["SaragaBeatDataset", "SaragaBeatDataset", "SaragaBeatDataset"]:
    """Split a track list into train / val / test datasets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(tracks)).tolist()
    n_train = int(len(tracks) * train_ratio)
    n_val   = int(len(tracks) * val_ratio)

    train_tracks = [tracks[i] for i in indices[:n_train]]
    val_tracks   = [tracks[i] for i in indices[n_train: n_train + n_val]]
    test_tracks  = [tracks[i] for i in indices[n_train + n_val:]]

    return (
        SaragaBeatDataset(train_tracks, **dataset_kwargs),
        SaragaBeatDataset(val_tracks,   **dataset_kwargs),
        SaragaBeatDataset(test_tracks,  **dataset_kwargs),
    )
