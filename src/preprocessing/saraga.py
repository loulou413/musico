"""Saraga Carnatic dataset loader via compiam / mirdata."""

import numpy as np
from pathlib import Path
from typing import Iterator

try:
    import compiam
except ImportError:
    compiam = None

try:
    import mirdata
except ImportError:
    mirdata = None

from .common import AudioTrack, load_audio, normalize


def load_saraga_track(track, data_home: str, sr: int = 22050) -> AudioTrack:
    """Load a single Saraga track into an AudioTrack object."""
    audio_path = str(track.audio_path)
    if audio_path.endswith(".mp3.mp3"):
        audio_path = audio_path[:-4]
    audio, _ = load_audio(audio_path, sr=sr)
    audio = normalize(audio)

    beat_times = None
    downbeat_times = None
    f0_times = None
    f0_freqs = None

    # Sama = first beat of each tala cycle; used as both beats and downbeats
    if hasattr(track, "sama") and track.sama is not None:
        beat_times = np.array(track.sama.times)
        downbeat_times = beat_times

    # Pitch (vocal / lead melodic line)
    if hasattr(track, "pitch_vocal") and track.pitch_vocal is not None:
        pv = track.pitch_vocal
        f0_times = np.array(pv.times)
        f0_freqs = np.array(pv.frequencies)

    metadata = {
        "tala": getattr(track, "tala", None),
        "raga": getattr(track, "raaga", None),
        "tempo": getattr(track, "tempo", None),
    }

    return AudioTrack(
        track_id=track.track_id,
        audio=audio,
        sr=sr,
        domain="carnatic",
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        f0_times=f0_times,
        f0_freqs=f0_freqs,
        metadata=metadata,
    )


def iter_saraga_tracks(data_home: str, sr: int = 22050) -> Iterator[AudioTrack]:
    """Yield all available Saraga Carnatic tracks."""
    if mirdata is None:
        raise ImportError("mirdata is required: pip install mirdata")

    dataset = mirdata.initialize("saraga_carnatic", data_home=data_home)
    for track_id in dataset.track_ids:
        track = dataset.track(track_id)
        try:
            yield load_saraga_track(track, data_home=data_home, sr=sr)
        except Exception as exc:
            print(f"[saraga] Skipping {track_id}: {exc}")


def get_saraga_split(
    data_home: str,
    split: str = "test",
    max_tracks: int | None = None,
    sr: int = 22050,
) -> list[AudioTrack]:
    """Return a list of AudioTrack objects for a given split."""
    if mirdata is None:
        raise ImportError("mirdata is required: pip install mirdata")

    dataset = mirdata.initialize("saraga_carnatic", data_home=data_home)
    dataset.download(partial_download=["index"])
    # mirdata provides a splits dict if the dataset defines one
    if hasattr(dataset, "get_track_ids_for_split"):
        ids = dataset.get_track_ids_for_split(split)
    else:
        ids = dataset.track_ids

    if max_tracks:
        ids = ids[:max_tracks]

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    tracks = []
    it = _tqdm(ids, desc="Loading Saraga", unit="track") if _tqdm else ids
    for tid in it:
        track = dataset.track(tid)
        try:
            tracks.append(load_saraga_track(track, data_home=data_home, sr=sr))
        except Exception as exc:
            msg = f"[saraga] Skipping {tid}: {exc}"
            if _tqdm:
                _tqdm.write(msg)
            else:
                print(msg)
    return tracks
