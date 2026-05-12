"""Cross-domain transfer experiments — Cassio's coordination.

Runs both beat and pitch models on both domains and produces a combined summary table.

Usage:
    python scripts/run_cross_domain.py \
        --saraga-home  data/raw/saraga \
        --gtzan-home   data/raw/gtzan \
        --maestro-home data/raw/maestro \
        --output-dir   results/cross_domain \
        --max-tracks   10
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.preprocessing.saraga import get_saraga_split
from src.preprocessing.gtzan import get_gtzan_split
from src.preprocessing.maestro import get_maestro_split
from src.cross_domain.experiments import (
    run_beat_cross_domain,
    run_pitch_cross_domain,
    summarise_cross_domain,
    save_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saraga-home", default="data/raw/saraga")
    parser.add_argument("--gtzan-home", default="data/raw/gtzan")
    parser.add_argument("--maestro-home", default="data/raw/maestro")
    parser.add_argument("--output-dir", default="results/cross_domain")
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    carnatic = get_saraga_split(args.saraga_home, split="test", max_tracks=args.max_tracks)
    western_rhythm = get_gtzan_split(args.gtzan_home, max_tracks=args.max_tracks)
    western_pitch = get_maestro_split(args.maestro_home, split="test", max_tracks=args.max_tracks)

    print("\n--- Beat tracking cross-domain ---")
    beat_df = run_beat_cross_domain(carnatic, western_rhythm)
    save_results(beat_df, str(out / "beat_cross_domain.csv"))
    beat_summary = summarise_cross_domain(beat_df, task="beat_tracking")
    print(beat_summary.to_string())

    print("\n--- Pitch estimation cross-domain ---")
    pitch_df = run_pitch_cross_domain(carnatic, western_pitch)
    save_results(pitch_df, str(out / "pitch_cross_domain.csv"))
    pitch_summary = summarise_cross_domain(pitch_df, task="pitch_estimation")
    print(pitch_summary.to_string())

    combined = pd.concat([beat_summary, pitch_summary])
    combined.to_csv(out / "cross_domain_summary.csv")
    print(f"\nCross-domain summary saved to {out / 'cross_domain_summary.csv'}")


if __name__ == "__main__":
    main()
