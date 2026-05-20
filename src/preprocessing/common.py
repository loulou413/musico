"""Shared audio loading and preprocessing utilities."""

import numpy as np
import librosa
import yaml
from pathlib import Path
from dataclasses import dataclass, field


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class AudioTrack:
    track_id: str
    audio: np.ndarray
    sr: int
    domain: str  # "carnatic" or "western"
    # Optional ground-truth annotations
    beat_times: np.ndarray | None = None       # seconds
    downbeat_times: np.ndarray | None = None   # seconds (sam / downbeat)
    f0_times: np.ndarray | None = None         # seconds
    f0_freqs: np.ndarray | None = None         # Hz (0 = unvoiced)
    f0_confidence: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def load_audio(path: str, sr: int = 22050, mono: bool = True) -> tuple[np.ndarray, int]:
    audio, sr_loaded = librosa.load(path, sr=sr, mono=mono)
    return audio, sr_loaded


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def hz_to_cents(freq_hz: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    """Convert Hz to cents relative to a reference frequency."""
    voiced = freq_hz > 0
    cents = np.zeros_like(freq_hz, dtype=float)
    cents[voiced] = 1200.0 * np.log2(freq_hz[voiced] / ref_hz)
    return cents


def cents_to_hz(cents: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    return ref_hz * (2.0 ** (cents / 1200.0))


def snap_to_semitone(freq_hz: np.ndarray) -> np.ndarray:
    """Snap each frequency to the nearest equal-temperament semitone (A=440 Hz)."""
    voiced = freq_hz > 0
    snapped = np.zeros_like(freq_hz)
    if voiced.any():
        midi = librosa.hz_to_midi(freq_hz[voiced])
        snapped[voiced] = librosa.midi_to_hz(np.round(midi))
    return snapped
