"""GuitarSet dataset loader (Western monophonic melody — used for pitch evaluation)."""

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

# Standard guitar string names in GuitarSet
_STRINGS = ["E", "A", "D", "G", "B", "e"]


def load_guitarset_track(track, sr: int = 22050) -> list[AudioTrack]:
    """Return one AudioTrack per active string (each is truly monophonic).

    Uses the hex_cln (hexaphonic clean) audio — one channel per string —
    so each returned track is a clean, monophonic signal with an accurate
    per-string F0 contour from the JAMS pitch_contour annotation.
    """
    import soundfile as sf

    if track.audio_hex_cln_path is None or not Path(track.audio_hex_cln_path).exists():
        return []

    # hex_cln is a 6-channel wav: channels map to E A D G B e
    audio_hex, sr_orig = sf.read(track.audio_hex_cln_path, always_2d=True)
    # audio_hex: (n_samples, 6)

    import librosa
    out_tracks = []

    pitch_contours = track.pitch_contours  # dict of {string: F0Data or None}
    if pitch_contours is None:
        return []

    for i, string in enumerate(_STRINGS):
        f0data = pitch_contours.get(string)
        if f0data is None:
            continue

        # Check that this string has any voiced frames
        freqs = np.array(f0data.frequencies)
        if not np.any(freqs > 0):
            continue

        # Extract and resample the single string channel
        mono = audio_hex[:, i].astype(np.float32)
        if sr_orig != sr:
            mono = librosa.resample(mono, orig_sr=sr_orig, target_sr=sr)
        mono = normalize(mono)

        f0_times = np.array(f0data.times)
        f0_freqs = np.array(f0data.frequencies)
        # mir_eval convention: 0 or negative = unvoiced
        f0_freqs = np.where(f0_freqs > 0, f0_freqs, 0.0)

        out_tracks.append(AudioTrack(
            track_id=f"{track.track_id}_{string}",
            audio=mono,
            sr=sr,
            domain="western",
            f0_times=f0_times,
            f0_freqs=f0_freqs,
            metadata={
                "style": track.style,
                "tempo": track.tempo,
                "mode": track.mode,
                "player_id": track.player_id,
                "string": string,
            },
        ))

    return out_tracks


def get_guitarset_tracks(
    data_home: str,
    mode: str = "solo",
    max_tracks: int | None = None,
    sr: int = 22050,
) -> list[AudioTrack]:
    """Load GuitarSet tracks as monophonic per-string AudioTrack objects.

    Args:
        data_home: path to GuitarSet data root
        mode: 'solo' or 'comp' (solo is cleaner for melody evaluation)
        max_tracks: cap on number of guitar tracks to load (before string expansion)
        sr: target sample rate
    """
    if mirdata is None:
        raise ImportError("mirdata is required: pip install mirdata")

    dataset = mirdata.initialize("guitarset", data_home=data_home)
    dataset.download(partial_download=["index"])
    ids = [tid for tid in dataset.track_ids if dataset.track(tid).mode == mode]
    if max_tracks:
        ids = ids[:max_tracks]

    out = []
    it = _tqdm(ids, desc="Loading GuitarSet", unit="track") if _tqdm else ids
    for tid in it:
        track = dataset.track(tid)
        try:
            string_tracks = load_guitarset_track(track, sr=sr)
            out.extend(string_tracks)
        except Exception as exc:
            msg = f"[guitarset] Skipping {tid}: {exc}"
            if _tqdm:
                _tqdm.write(msg)
            else:
                print(msg)

    return out
