"""Cross-domain aggregation — Cassio's task.

Reads the per-condition result files produced by Clara (rhythm) and Louis
(pitch) and stitches them into a single unified summary table covering
all 3 conditions (A / B / C) × 2 domains (Western / Carnatic) for each
task.

This script does NOT retrain or re-run any model. It only reads CSVs and
JSONs from disk, so it's cheap (milliseconds) and can be re-run any time
new results land.

Expected input layout (see TUTORIAL.md §3 and §4):

    results/
    ├── rhythm/
    │   ├── A_madmom/beat_results.csv
    │   ├── B_carnatic_from_scratch/test_summary.json
    │   ├── B_on_western/beat_results.csv
    │   ├── C_finetuned_from_western/test_summary.json
    │   └── C_on_western/beat_results.csv
    └── pitch/
        ├── A_crepe/pitch_results.csv
        ├── B_carnatic_from_scratch/comparison_all.csv
        └── C_finetuned_from_crepe/comparison_all.csv

Missing files emit a warning but don't crash — useful while teammates'
runs are still in progress.

Usage:
    PYTHONPATH=. python scripts/run_cross_domain.py \
        --results-root results \
        --output-dir   results/cross_domain
"""

import argparse
from pathlib import Path

from src.cross_domain.experiments import (
    load_all_results,
    summarise,
    pivot_table,
    save_results,
    discover_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results",
                        help="Root directory containing rhythm/ and pitch/ subfolders")
    parser.add_argument("--output-dir", default="results/cross_domain")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any expected result file is missing")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Report which files were found ─────────────────────────────────────
    print("Discovering result files…")
    found = discover_results(results_root)
    for label, path in found.items():
        status = "✓" if path else "MISSING"
        print(f"  [{status}] {label:<10} {path if path else ''}")

    # ── Load + aggregate ──────────────────────────────────────────────────
    print("\nLoading all available results…")
    long_df = load_all_results(results_root, strict=args.strict)
    save_results(long_df, str(out / "all_results_long.csv"))

    summary = summarise(long_df)
    save_results(summary, str(out / "cross_domain_summary.csv"))

    # ── Pretty-print the 3 × 2 tables ─────────────────────────────────────
    print("\n=== 3 × 2 summary tables (mean across tracks) ===")
    for task, metric in [
        ("beat",  "f_measure"),
        ("beat",  "cemgil"),
        ("beat",  "downbeat_f_measure"),
        ("pitch", "raw_pitch_accuracy"),
        ("pitch", "overall_accuracy"),
        ("pitch", "mean_abs_error_cents"),
    ]:
        table = pivot_table(summary, metric=metric, task=task)
        if table.empty:
            continue
        print(f"\n--- {task} | {metric} ---")
        print(table.to_string())

    print(f"\nSaved long-form table:  {out / 'all_results_long.csv'}")
    print(f"Saved summary table:    {out / 'cross_domain_summary.csv'}")
    print("\nNext: open notebooks/04_cross_domain.ipynb for figures.")


if __name__ == "__main__":
    main()
