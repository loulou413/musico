"""Evaluate a trained Carnatic pitch model without retraining.

Loads an existing best_model.pt checkpoint and runs the full --compare-all
evaluation: 3 methods (carnatic, crepe, pyin) × 2 domains (carnatic, western).

Usage:
    PYTHONPATH=. python scripts/eval_pitch_carnatic.py \
        --checkpoint      results/pitch/B_carnatic_from_scratch/best_model.pt \
        --saraga-home     data/raw/saraga \
        --guitarset-home  data/raw/guitarset \
        --output-dir      results/pitch/B_carnatic_from_scratch \
        --max-tracks      60
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.saraga import get_saraga_split
from src.preprocessing.guitarset import get_guitarset_tracks
from src.pitch.estimation import run_pitch_estimation
from src.pitch.evaluation import evaluate_pitch, aggregate_results
from src.pitch.analysis import plot_metric_by_domain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--saraga-home", default="data/raw/saraga")
    parser.add_argument("--guitarset-home", default="data/raw/guitarset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # ── Load all Saraga tracks ────────────────────────────────────────────────
    print("Loading Saraga tracks…")
    import mirdata
    from src.preprocessing.saraga import load_saraga_track
    ds = mirdata.initialize("saraga_carnatic", data_home=args.saraga_home)
    all_tracks = []
    for tid in ds.track_ids[: args.max_tracks]:
        try:
            all_tracks.append(load_saraga_track(ds.track(tid), args.saraga_home))
        except Exception as e:
            print(f"  Skip {tid}: {e}")
    print(f"  Loaded {len(all_tracks)} Saraga tracks")

    test_tracks_carnatic = all_tracks
    print(f"  Evaluating on all {len(test_tracks_carnatic)} tracks")

    # ── Load GuitarSet ────────────────────────────────────────────────────────
    print("\nLoading GuitarSet tracks…")
    western_tracks = get_guitarset_tracks(
        args.guitarset_home, mode="solo", max_tracks=args.max_tracks
    )
    print(f"  Loaded {len(western_tracks)} GuitarSet string-tracks")

    # ── Run 3 methods × 2 domains (with resume support) ──────────────────────
    methods = [
        ("carnatic", {"carnatic_checkpoint": checkpoint}),
        ("crepe",    {}),
        ("pyin",     {}),
    ]

    checkpoint_csv = out / "comparison_all.csv"
    if checkpoint_csv.exists():
        existing = pd.read_csv(checkpoint_csv)
        all_rows = existing.to_dict("records")
        done = set(zip(existing["method"], existing["track_id"]))
        print(f"  Resuming from checkpoint — {len(all_rows)} results already saved")
    else:
        all_rows = []
        done = set()

    for method, extra_kwargs in methods:
        print(f"\n--- {method} ---")
        for track in test_tracks_carnatic + western_tracks:
            if (method, track.track_id) in done:
                continue
            try:
                pred = run_pitch_estimation(track, method=method, **extra_kwargs)
                result = evaluate_pitch(track, pred)
                row = {"method": method, **result.__dict__}
                all_rows.append(row)
                done.add((method, track.track_id))
                print(f"  [{track.domain}] {track.track_id}  RPA={result.raw_pitch_accuracy:.3f}")
                # save incrementally
                pd.DataFrame(all_rows).to_csv(checkpoint_csv, index=False)
            except Exception as e:
                print(f"  Skip {track.track_id} ({method}): {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(checkpoint_csv, index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== RPA by method × domain ===")
    summary = df.groupby(["method", "domain"])["raw_pitch_accuracy"].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    summary.to_csv(out / "comparison_summary.csv")

    print("\n=== Overall Accuracy by method × domain ===")
    oa_summary = df.groupby(["method", "domain"])["overall_accuracy"].agg(["mean", "std"]).round(4)
    print(oa_summary.to_string())

    print("\n=== Mean Abs Error (cents) by method × domain ===")
    mae_summary = df.groupby(["method", "domain"])["mean_abs_error_cents"].agg(["mean", "std"]).round(1)
    print(mae_summary.to_string())

    # ── Figures ───────────────────────────────────────────────────────────────
    for metric in ["raw_pitch_accuracy", "overall_accuracy", "mean_abs_error_cents", "semitone_snap_ratio"]:
        plot_metric_by_domain(df, metric, str(out / "figures"))

    # ── Per-domain JSON summaries ─────────────────────────────────────────────
    for method in df["method"].unique():
        for domain in df["domain"].unique():
            subset = df[(df["method"] == method) & (df["domain"] == domain)]
            if subset.empty:
                continue
            s = {
                col: {"mean": float(subset[col].mean()), "std": float(subset[col].std())}
                for col in ["raw_pitch_accuracy", "overall_accuracy",
                            "mean_abs_error_cents", "voicing_recall"]
            }
            fname = out / f"summary_{method}_{domain}.json"
            with open(fname, "w") as f:
                json.dump(s, f, indent=2)

    print(f"\nAll outputs saved to {out}")


if __name__ == "__main__":
    main()
