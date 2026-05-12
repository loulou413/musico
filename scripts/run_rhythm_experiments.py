"""Beat tracking experiments — Clara's task.

Evaluates madmom DBNBeatTracker on Saraga (Carnatic) and GTZAN (Western).
Produces per-track results CSVs and summary plots.

Usage:
    python scripts/run_rhythm_experiments.py \
        --saraga-home data/raw/saraga \
        --gtzan-home  data/raw/gtzan \
        --output-dir  results/rhythm \
        --max-tracks  20
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.preprocessing.saraga import get_saraga_split
from src.preprocessing.gtzan import get_gtzan_split
from src.rhythm.beat_tracking import run_beat_tracking
from src.rhythm.evaluation import evaluate_beats, aggregate_results
from src.rhythm.analysis import (
    results_to_dataframe,
    plot_metric_by_domain,
    plot_tempo_accuracy,
    drift_analysis,
    plot_drift,
    sam_alignment_analysis,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saraga-home", default="data/raw/saraga")
    parser.add_argument("--gtzan-home", default="data/raw/gtzan")
    parser.add_argument("--output-dir", default="results/rhythm")
    parser.add_argument("--backend", default="madmom", choices=["madmom", "librosa", "carnatic"])
    parser.add_argument("--carnatic-checkpoint", default=None,
                        help="Path to trained Saraga model checkpoint (required when --backend=carnatic)")
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    if args.backend == "carnatic" and args.carnatic_checkpoint is None:
        parser.error("--carnatic-checkpoint is required when --backend=carnatic")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Saraga tracks...")
    carnatic = get_saraga_split(args.saraga_home, split="test", max_tracks=args.max_tracks)

    print("Loading GTZAN tracks...")
    western = get_gtzan_split(args.gtzan_home, max_tracks=args.max_tracks)

    all_results = []
    for track in carnatic + western:
        print(f"  [{track.domain}] {track.track_id}")
        pred = run_beat_tracking(
            track,
            backend=args.backend,
            carnatic_checkpoint=args.carnatic_checkpoint,
        )
        result = evaluate_beats(track, pred)
        all_results.append(result)

        # Per-track drift plot
        if track.beat_times is not None and len(track.beat_times) > 0:
            ref_t, errors = drift_analysis(track.beat_times, pred.beat_times)
            plot_drift(ref_t, errors, track.track_id, str(out / "drift"))

        # Sam alignment (Carnatic only)
        if track.domain == "carnatic" and track.downbeat_times is not None:
            sam = sam_alignment_analysis(track.downbeat_times, pred.downbeat_times)
            print(f"    Sam alignment: {sam}")

    df = results_to_dataframe(all_results)
    df.to_csv(out / "beat_results.csv", index=False)

    summary = aggregate_results(all_results)
    with open(out / "beat_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    for metric in ["f_measure", "cemgil", "continuity", "downbeat_f_measure"]:
        plot_metric_by_domain(df, metric, str(out / "figures"))
    plot_tempo_accuracy(df, str(out / "figures"))

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
