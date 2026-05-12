"""Beat tracking wrappers: madmom DBNBeatTracker, librosa fallback, and trained Carnatic model."""

import numpy as np
import librosa
from dataclasses import dataclass
from pathlib import Path

try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    _MADMOM_AVAILABLE = True
except ImportError:
    _MADMOM_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.preprocessing.common import AudioTrack


@dataclass
class BeatPrediction:
    track_id: str
    beat_times: np.ndarray       # predicted beat positions (seconds)
    downbeat_times: np.ndarray   # predicted downbeat / sam positions (seconds)
    bpm: float


def track_beats_madmom(audio: np.ndarray, sr: int, fps: int = 100) -> np.ndarray:
    """Run madmom DBNBeatTracker on a waveform. Returns beat times in seconds."""
    if not _MADMOM_AVAILABLE:
        raise ImportError("madmom is required: pip install madmom")
    proc = RNNBeatProcessor(fps=fps)
    beat_act = proc(audio.astype(np.float32))
    dbn = DBNBeatTrackingProcessor(fps=fps)
    beat_times = dbn(beat_act)
    return beat_times


def track_beats_librosa(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """librosa beat tracker (fallback). Returns beat times in seconds."""
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    return librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)


def estimate_downbeats(beat_times: np.ndarray, beats_per_bar: int = 4) -> np.ndarray:
    """Naive downbeat estimation: every N-th beat.
    For Carnatic music this is used as a baseline; sam detection is separate."""
    return beat_times[::beats_per_bar]


def track_beats_carnatic_model(
    audio: np.ndarray,
    sr: int,
    checkpoint_path: str,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    """Run the Saraga-trained beat activation model. Returns beat times in seconds."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required: pip install torch")

    from .dataset import compute_mel
    from .model import BeatActivationModel, activation_to_beat_times
    from .train import load_model

    model = load_model(checkpoint_path, n_mels=n_mels)
    mel = compute_mel(audio, sr, hop_length=hop_length, n_mels=n_mels)  # (n_mels, T)
    mel_t = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)

    with torch.no_grad():
        activation = model(mel_t).squeeze(0)  # (T,)

    return activation_to_beat_times(activation, sr=sr, hop_length=hop_length)


def run_beat_tracking(
    track: AudioTrack,
    backend: str = "madmom",
    beats_per_bar: int = 4,
    carnatic_checkpoint: str | None = None,
) -> BeatPrediction:
    """Run beat tracking with the selected backend.

    backend options:
        "madmom"   — pre-trained Western DBNBeatTracker (default)
        "librosa"  — DSP onset-based (fallback)
        "carnatic" — Saraga-trained CNN+BiLSTM model (requires carnatic_checkpoint)
    """
    if backend == "carnatic":
        if carnatic_checkpoint is None:
            raise ValueError("carnatic_checkpoint path is required for backend='carnatic'")
        beat_times = track_beats_carnatic_model(
            track.audio, track.sr, checkpoint_path=carnatic_checkpoint
        )
    elif backend == "madmom" and _MADMOM_AVAILABLE:
        beat_times = track_beats_madmom(track.audio, track.sr)
    else:
        beat_times = track_beats_librosa(track.audio, track.sr)

    downbeat_times = estimate_downbeats(beat_times, beats_per_bar=beats_per_bar)
    bpm = 60.0 / np.median(np.diff(beat_times)) if len(beat_times) > 1 else 0.0

    return BeatPrediction(
        track_id=track.track_id,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        bpm=bpm,
    )
