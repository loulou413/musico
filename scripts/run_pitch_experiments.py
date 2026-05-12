"""Pitch estimation experiments — Louis's task.

Evaluates CREPE (and pyin baseline) on Saraga (Carnatic) and MAESTRO (Western).
Includes gamaka detection and semitone-snapping bias analysis.

Usage:
    python scripts/run_pitch_experiments.py \
        --saraga-home  data/raw/saraga \
        --maestro-home data/raw/maestro \
        --output-dir   results/pitch \
        --method       crepe \
        --max-tracks   10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.saraga import get_saraga_split
from src.preprocessing.maestro import get_maestro_split
from src.pitch.estimation import run_pitch_estimation
from src.pitch.evaluation import evaluate_pitch, aggregate_results
from src.pitch.analysis import (
    results_to_dataframe,
    plot_metric_by_domain,
    plot_cents_error_histogram,
    detect_gamaka_regions,
    plot_pitch_contour,
    gamaka_error_vs_non_gamaka,
)
from src.preprocessing.common import hz_to_cents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saraga-home", default="data/raw/saraga")
    parser.add_argument("--maestro-home", default="data/raw/maestro")
    parser.add_argument("--output-dir", default="results/pitch")
    parser.add_argument("--method", default="crepe", choices=["crepe", "pyin", "carnatic"])
    parser.add_argument("--carnatic-checkpoint", default=None,
                        help="Path to trained Saraga model checkpoint (required when --method=carnatic)")
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    if args.method == "carnatic" and args.carnatic_checkpoint is None:
        parser.error("--carnatic-checkpoint is required when --method=carnatic")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Saraga tracks...")
    carnatic = get_saraga_split(args.saraga_home, split="test", max_tracks=args.max_tracks)

    print("Loading MAESTRO tracks...")
    western = get_maestro_split(args.maestro_home, split="test", max_tracks=args.max_tracks)

    all_results = []
    gamaka_errors = []

    for track in carnatic + western:
        print(f"  [{track.domain}] {track.track_id}")
        pred = run_pitch_estimation(
            track, method=args.method,
            carnatic_checkpoint=args.carnatic_checkpoint,
        )
        result = evaluate_pitch(track, pred)
        all_results.append(result)

        if track.f0_times is not None and track.f0_freqs is not None:
            # Cents error histogram
            plot_cents_error_histogram(
                track.f0_freqs, pred.frequencies,
                domain=track.domain, output_dir=str(out / "figures"),
            )

            # Gamaka analysis (Carnatic only)
            if track.domain == "carnatic":
                gamaka_mask = detect_gamaka_regions(track.f0_times, track.f0_freqs)
                plot_pitch_contour(
                    track.f0_times, track.f0_freqs, pred.frequencies,
                    gamaka_mask=gamaka_mask,
                    track_id=track.track_id,
                    output_dir=str(out / "figures"),
                )
                g_err = gamaka_error_vs_non_gamaka(
                    track.f0_freqs, pred.frequencies, gamaka_mask
                )
                g_err["track_id"] = track.track_id
                gamaka_errors.append(g_err)
                print(f"    Gamaka MAE={g_err['mae_gamaka']:.1f}c  non-gamaka MAE={g_err['mae_non_gamaka']:.1f}c")

    df = results_to_dataframe(all_results)
    df.to_csv(out / "pitch_results.csv", index=False)

    summary = aggregate_results(all_results)
    with open(out / "pitch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    for metric in ["raw_pitch_accuracy", "raw_chroma_accuracy", "overall_accuracy",
                   "mean_abs_error_cents", "semitone_snap_ratio"]:
        plot_metric_by_domain(df, metric, str(out / "figures"))

    if gamaka_errors:
        pd.DataFrame(gamaka_errors).to_csv(out / "gamaka_errors.csv", index=False)

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
