"""GTZAN genre dataset loader (used for rhythm / beat tracking evaluation)."""

import numpy as np
from pathlib import Path
from typing import Iterator

try:
    import mirdata
except ImportError:
    mirdata = None

from .common import AudioTrack, load_audio, normalize


def load_gtzan_track(track, sr: int = 22050) -> AudioTrack:
    audio_path = track.audio_path
    audio, _ = load_audio(audio_path, sr=sr)
    audio = normalize(audio)

    beat_times = None
    if hasattr(track, "beats") and track.beats is not None:
        beat_times = np.array(track.beats.beat_times)

    metadata = {
        "genre": getattr(track, "genre", None),
    }

    return AudioTrack(
        track_id=track.track_id,
        audio=audio,
        sr=sr,
        domain="western",
        beat_times=beat_times,
        metadata=metadata,
    )


def get_gtzan_split(
    data_home: str,
    genres: list[str] | None = None,
    max_tracks: int | None = None,
    sr: int = 22050,
) -> list[AudioTrack]:
    """Return GTZAN tracks, optionally filtered by genre list."""
    if mirdata is None:
        raise ImportError("mirdata is required: pip install mirdata")

    dataset = mirdata.initialize("gtzan_genre", data_home=data_home)
    ids = dataset.track_ids

    if genres:
        ids = [tid for tid in ids if dataset.track(tid).genre in genres]
    if max_tracks:
        ids = ids[:max_tracks]

    tracks = []
    for tid in ids:
        track = dataset.track(tid)
        try:
            tracks.append(load_gtzan_track(track, sr=sr))
        except Exception as exc:
            print(f"[gtzan] Skipping {tid}: {exc}")
    return tracks
