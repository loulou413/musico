"""Pitch estimation wrappers: CREPE, pyin, and trained Carnatic model."""

import numpy as np
import librosa
from dataclasses import dataclass
from pathlib import Path

try:
    import torchcrepe
    _CREPE_AVAILABLE = True
except ImportError:
    _CREPE_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.preprocessing.common import AudioTrack


@dataclass
class PitchPrediction:
    track_id: str
    times: np.ndarray           # seconds
    frequencies: np.ndarray     # Hz (0 = unvoiced)
    confidence: np.ndarray      # [0, 1]
    method: str                 # "crepe" | "pyin"


def estimate_pitch_crepe(
    audio: np.ndarray,
    sr: int,
    model_capacity: str = "full",
    viterbi: bool = True,
    confidence_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (times_s, freqs_hz, confidence). Unvoiced frames → freq=0."""
    if not _CREPE_AVAILABLE:
        raise ImportError("torchcrepe is required: pip install torchcrepe")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    hop_length = int(sr * 0.01)
    freqs, conf = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        fmin=32.70,
        fmax=1975.5,
        model=model_capacity,
        decoder=torchcrepe.decode.viterbi if viterbi else torchcrepe.decode.argmax,
        device=device,
        return_periodicity=True,
        batch_size=2048,
    )
    freqs = freqs.squeeze(0).cpu().numpy()
    conf = conf.squeeze(0).cpu().numpy()
    times = np.arange(len(freqs)) * 0.01
    freqs[conf < confidence_threshold] = 0.0
    return times, freqs, conf


def estimate_pitch_pyin(
    audio: np.ndarray,
    sr: int,
    fmin: float = librosa.note_to_hz("C2"),
    fmax: float = librosa.note_to_hz("C7"),
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (times_s, freqs_hz, voiced_flag as float)."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~voiced_flag] = 0.0
    return times, f0, voiced_flag.astype(float)


def estimate_pitch_carnatic_model(
    audio: np.ndarray,
    sr: int,
    checkpoint_path: str,
    viterbi: bool = True,
    confidence_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the Saraga-trained CREPELike model. Returns (times_s, freqs_hz, confidence)."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required: pip install torch")

    from .dataset import (
        _resample, _normalize_frame, TARGET_SR, FRAME_LEN, HOP_SAMPLES, N_BINS
    )
    from .model import CREPELike, activation_to_hz, activation_to_hz_viterbi
    from .train import load_model

    model = load_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    audio16 = _resample(audio, sr)
    half = FRAME_LEN // 2
    audio_padded = np.pad(audio16, (half, half))

    starts = np.arange(0, len(audio16), HOP_SAMPLES)
    times = starts / TARGET_SR

    activations = []
    batch_size = 256
    frames_batch = []

    for s in starts:
        frame = audio_padded[s: s + FRAME_LEN]
        if len(frame) < FRAME_LEN:
            frame = np.pad(frame, (0, FRAME_LEN - len(frame)))
        frames_batch.append(_normalize_frame(frame))

        if len(frames_batch) == batch_size:
            activations.append(_run_batch(model, frames_batch, device))
            frames_batch = []

    if frames_batch:
        activations.append(_run_batch(model, frames_batch, device))

    act_all = np.concatenate(activations, axis=0)
    confidence = act_all.max(axis=1)

    if viterbi:
        freqs = activation_to_hz_viterbi(act_all)
    else:
        freqs = np.array([activation_to_hz(a) for a in act_all])

    freqs[confidence < confidence_threshold] = 0.0

    return times[: len(freqs)], freqs, confidence


def _run_batch(model, frames: list, device) -> np.ndarray:
    import torch
    x = torch.tensor(np.stack(frames), dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()


def run_pitch_estimation(
    track: AudioTrack,
    method: str = "crepe",
    carnatic_checkpoint: str | None = None,
    **kwargs,
) -> PitchPrediction:
    """Run pitch estimation with the selected method.

    method options:
        "crepe"    — pre-trained Western CREPE model (default)
        "pyin"     — DSP baseline (librosa)
        "carnatic" — Saraga-trained CREPELike model (requires carnatic_checkpoint)
    """
    if method == "carnatic":
        if carnatic_checkpoint is None:
            raise ValueError("carnatic_checkpoint path is required for method='carnatic'")
        times, freqs, conf = estimate_pitch_carnatic_model(
            track.audio, track.sr, checkpoint_path=carnatic_checkpoint, **kwargs
        )
    elif method == "crepe":
        times, freqs, conf = estimate_pitch_crepe(track.audio, track.sr, **kwargs)
    elif method == "pyin":
        times, freqs, conf = estimate_pitch_pyin(track.audio, track.sr, **kwargs)
    else:
        raise ValueError(f"Unknown pitch method: {method}")

    return PitchPrediction(
        track_id=track.track_id,
        times=times,
        frequencies=freqs,
        confidence=conf,
        method=method,
    )
