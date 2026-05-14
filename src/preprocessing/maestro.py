"""MAESTRO dataset loader (piano — used for pitch evaluation)."""

import numpy as np
from pathlib import Path
from typing import Iterator

try:
    import mirdata
except ImportError:
    mirdata = None

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .common import AudioTrack, load_audio, normalize


def load_maestro_track(track, sr: int = 22050) -> AudioTrack:
    audio_path = track.audio_path
    audio, _ = load_audio(audio_path, sr=sr)
    audio = normalize(audio)

    # MAESTRO provides note annotations; derive a rough f0 from the dominant note
    # For melody evaluation we keep it None and rely on CREPE estimates
    f0_times, f0_freqs = None, None
    if hasattr(track, "notes") and track.notes is not None:
        notes = track.notes
        # Build a frame-level pitch annotation at 10 ms resolution.
        # Only keep frames where exactly one note is active — polyphonic frames
        # are set to 0 (unvoiced) so mir_eval only scores unambiguous pitches.
        duration = audio.shape[0] / sr
        times = np.arange(0, duration, 0.01)
        note_count = np.zeros(len(times), dtype=int)
        freqs = np.zeros(len(times))
        starts = notes.intervals[:, 0] if hasattr(notes, "intervals") else notes.start_times
        ends   = notes.intervals[:, 1] if hasattr(notes, "intervals") else notes.end_times
        for onset, offset, pitch in zip(starts, ends, notes.pitches):
            mask = (times >= onset) & (times < offset)
            midi_hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            note_count[mask] += 1
            freqs[mask] = np.maximum(freqs[mask], midi_hz)
        # Zero out polyphonic frames
        freqs[note_count != 1] = 0.0
        f0_times = times
        f0_freqs = freqs

    metadata = {
        "composer": getattr(track, "composer", None),
        "title": getattr(track, "title", None),
        "year": getattr(track, "year", None),
    }

    return AudioTrack(
        track_id=track.track_id,
        audio=audio,
        sr=sr,
        domain="western",
        f0_times=f0_times,
        f0_freqs=f0_freqs,
        metadata=metadata,
    )


def get_maestro_split(
    data_home: str,
    split: str = "test",
    max_tracks: int | None = None,
    sr: int = 22050,
) -> list[AudioTrack]:
    if mirdata is None:
        raise ImportError("mirdata is required: pip install mirdata")

    dataset = mirdata.initialize("maestro", data_home=data_home)
    ids = [
        tid
        for tid in dataset.track_ids
        if dataset.track(tid).split == split
    ]
    if max_tracks:
        ids = ids[:max_tracks]

    tracks = []
    it = _tqdm(ids, desc="Loading MAESTRO", unit="track") if _tqdm else ids
    for tid in it:
        track = dataset.track(tid)
        try:
            tracks.append(load_maestro_track(track, sr=sr))
        except Exception as exc:
            msg = f"[maestro] Skipping {tid}: {exc}"
            if _tqdm:
                _tqdm.write(msg)
            else:
                print(msg)
    return tracks
