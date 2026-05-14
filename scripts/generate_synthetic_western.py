"""Generate a small synthetic Western pitch dataset for CREPE validation.

Produces clean monophonic audio (band-limited sawtooth waves) at known MIDI pitches,
with perfect frame-level F0 ground truth. Saves to data/raw/synthetic_western/.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import soundfile as sf


SR = 22050
HOP_S = 0.01          # 10 ms frames (matches CREPE)
NOTE_DUR_S = 1.0      # each note lasts 1 second
FADE_MS = 20          # fade in/out to avoid clicks


def sawtooth_note(midi_note: int, duration_s: float, sr: int) -> np.ndarray:
    """Band-limited sawtooth wave for a single MIDI note."""
    freq = 440.0 * 2.0 ** ((midi_note - 69) / 12.0)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    # Sum harmonics up to Nyquist
    wave = np.zeros(n)
    k = 1
    while freq * k < sr / 2:
        wave += ((-1) ** (k + 1)) / k * np.sin(2 * np.pi * freq * k * t)
        k += 1
    wave *= 2 / np.pi  # normalise to [-1, 1]
    # Fade in/out
    fade = int(FADE_MS * sr / 1000)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return wave.astype(np.float32)


def build_track(midi_notes: list[int], sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate notes into one audio track with matching f0 annotation."""
    segments, f0_times, f0_freqs = [], [], []
    t = 0.0
    for midi in midi_notes:
        audio = sawtooth_note(midi, NOTE_DUR_S, sr)
        segments.append(audio)
        freq_hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        frames = np.arange(0, NOTE_DUR_S, HOP_S)
        f0_times.extend(t + frames)
        f0_freqs.extend([freq_hz] * len(frames))
        t += NOTE_DUR_S
    return (
        np.concatenate(segments),
        np.array(f0_times),
        np.array(f0_freqs),
    )


# Western scale patterns (MIDI note numbers, C4=60)
TRACKS = {
    "chromatic_c4_c5": list(range(60, 73)),          # C4 → C5 chromatic
    "major_scale_c4":  [60, 62, 64, 65, 67, 69, 71, 72],
    "jazz_intervals":  [55, 62, 67, 71, 74, 67, 62, 55],  # mixed register
    "low_register":    [36, 38, 40, 41, 43, 45, 47, 48],  # C2 → C3
    "high_register":   [72, 74, 76, 77, 79, 81, 83, 84],  # C5 → C6
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/synthetic_western")
    parser.add_argument("--sr", type=int, default=SR)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index = {}
    for name, notes in TRACKS.items():
        audio, f0_times, f0_freqs = build_track(notes, sr=args.sr)
        audio_path = out / f"{name}.wav"
        annot_path = out / f"{name}.json"
        sf.write(str(audio_path), audio, args.sr)
        with open(annot_path, "w") as f:
            json.dump({"f0_times": f0_times.tolist(), "f0_freqs": f0_freqs.tolist()}, f)
        index[name] = {"audio": str(audio_path), "annotation": str(annot_path)}
        print(f"  {name}: {len(notes)} notes, {len(audio)/args.sr:.1f}s")

    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved {len(TRACKS)} tracks to {out}")


if __name__ == "__main__":
    main()
