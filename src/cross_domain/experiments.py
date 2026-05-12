"""Cross-domain aggregation: read Clara's and Louis's per-condition result
files from disk and stitch them into one unified table.

This module does NOT retrain or re-run any model. Its job is purely to:
  1. discover which result files exist on disk
  2. tag each row with (task, condition, domain)
  3. concatenate everything into a single long-form DataFrame
  4. produce the 3 × 2 summary table per task and per metric

The cross-domain script is therefore cheap to run (milliseconds) and can
be re-run any time a teammate produces new results.

Layout expected on disk (matches TUTORIAL.md sections 3 and 4):

    results/
    ├── rhythm/
    │   ├── A_madmom/beat_results.csv                              ← A-W + A-C
    │   ├── B_carnatic_from_scratch/test_summary.json              ← B-C (aggregate only)
    │   ├── B_on_western/beat_results.csv                          ← B-W
    │   ├── C_finetuned_from_western/test_summary.json             ← C-C (aggregate only)
    │   └── C_on_western/beat_results.csv                          ← C-W
    └── pitch/
        ├── A_crepe/pitch_results.csv                              ← A-W + A-C
        ├── B_carnatic_from_scratch/comparison_all.csv             ← B-W + B-C
        └── C_finetuned_from_crepe/comparison_all.csv              ← C-W + C-C

A per-condition result file is OPTIONAL — if it doesn't exist yet, the
aggregator just leaves those rows blank in the summary and emits a
warning so you can see which teammates haven't finished.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ── Schema ────────────────────────────────────────────────────────────────────

# Beat metrics we promote to the summary table
BEAT_METRICS = ["f_measure", "cemgil", "continuity", "downbeat_f_measure"]
# Pitch metrics we promote to the summary table
PITCH_METRICS = ["raw_pitch_accuracy", "overall_accuracy", "mean_abs_error_cents"]

# Long-form schema: every row is one (track, condition, domain, metric, value).
LONG_COLS = ["task", "condition", "domain", "track_id", "metric", "value"]


# ── Beat loaders ──────────────────────────────────────────────────────────────

def _load_beat_csv(path: Path, condition: str) -> pd.DataFrame:
    """Per-track beat CSV produced by run_rhythm_experiments.py.

    Columns: track_id, domain, f_measure, cemgil, continuity, ...
    """
    df = pd.read_csv(path)
    df = df.melt(
        id_vars=["track_id", "domain"],
        value_vars=[c for c in BEAT_METRICS if c in df.columns],
        var_name="metric",
        value_name="value",
    )
    df["task"] = "beat"
    df["condition"] = condition
    return df[LONG_COLS]


def _load_beat_summary_json(path: Path, condition: str, domain: str) -> pd.DataFrame:
    """test_summary.json — aggregate-only file (no per-track rows).

    We emit one synthetic row per metric with track_id='__aggregate__' and
    `value` = the reported mean. std is preserved separately in the
    summary step (see _aggregate_beat). This is a fallback when the
    per-track CSV isn't available.
    """
    with open(path) as f:
        agg = json.load(f)
    rows = []
    for m in BEAT_METRICS:
        if m in agg and "mean" in agg[m] and not np.isnan(agg[m]["mean"]):
            rows.append({
                "task": "beat",
                "condition": condition,
                "domain": domain,
                "track_id": "__aggregate__",
                "metric": m,
                "value": agg[m]["mean"],
            })
    return pd.DataFrame(rows, columns=LONG_COLS)


# ── Pitch loaders ─────────────────────────────────────────────────────────────

def _load_pitch_csv(
    path: Path, condition: str, method_filter: str | None = None
) -> pd.DataFrame:
    """Per-track pitch CSV.

    Two flavours:
    - run_pitch_experiments.py output: columns include track_id, domain,
      raw_pitch_accuracy, etc. (no 'method' column).
    - train_pitch_carnatic.py --compare-all output: also has a 'method'
      column (one of 'carnatic', 'crepe', 'pyin'). When method_filter
      is set, only rows matching it are kept.
    """
    df = pd.read_csv(path)
    if "method" in df.columns and method_filter is not None:
        df = df[df["method"] == method_filter].copy()
    df = df.melt(
        id_vars=["track_id", "domain"],
        value_vars=[c for c in PITCH_METRICS if c in df.columns],
        var_name="metric",
        value_name="value",
    )
    df["task"] = "pitch"
    df["condition"] = condition
    return df[LONG_COLS]


# ── Top-level: discover and load everything ───────────────────────────────────

def discover_results(results_root: Path) -> dict[str, Path | None]:
    """Return a dict {label: path_or_None} for every result file we know about."""
    r = results_root
    expected = {
        # condition: file
        "beat_A":   r / "rhythm" / "A_madmom" / "beat_results.csv",
        "beat_B_W": r / "rhythm" / "B_on_western" / "beat_results.csv",
        "beat_B_C": r / "rhythm" / "B_carnatic_from_scratch" / "test_summary.json",
        "beat_C_W": r / "rhythm" / "C_on_western" / "beat_results.csv",
        "beat_C_C": r / "rhythm" / "C_finetuned_from_western" / "test_summary.json",
        "pitch_A":  r / "pitch" / "A_crepe" / "pitch_results.csv",
        "pitch_B":  r / "pitch" / "B_carnatic_from_scratch" / "comparison_all.csv",
        "pitch_C":  r / "pitch" / "C_finetuned_from_crepe" / "comparison_all.csv",
    }
    return {label: (p if p.exists() else None) for label, p in expected.items()}


def load_all_results(results_root: Path, strict: bool = False) -> pd.DataFrame:
    """Load every available per-condition result file as a long-form DataFrame.

    If `strict=True`, raise FileNotFoundError when any expected file is missing.
    If `strict=False` (default), warn and continue with what's there.
    """
    paths = discover_results(results_root)
    frames: list[pd.DataFrame] = []

    # ── Beat ──────────────────────────────────────────────────────────────
    if paths["beat_A"]:
        frames.append(_load_beat_csv(paths["beat_A"], "A"))
    if paths["beat_B_W"]:
        frames.append(_load_beat_csv(paths["beat_B_W"], "B"))
    if paths["beat_B_C"]:
        frames.append(_load_beat_summary_json(paths["beat_B_C"], "B", "carnatic"))
    if paths["beat_C_W"]:
        frames.append(_load_beat_csv(paths["beat_C_W"], "C"))
    if paths["beat_C_C"]:
        frames.append(_load_beat_summary_json(paths["beat_C_C"], "C", "carnatic"))

    # ── Pitch ─────────────────────────────────────────────────────────────
    if paths["pitch_A"]:
        frames.append(_load_pitch_csv(paths["pitch_A"], "A"))
    if paths["pitch_B"]:
        # comparison_all.csv contains the Carnatic-trained model under method='carnatic'
        frames.append(_load_pitch_csv(paths["pitch_B"], "B", method_filter="carnatic"))
    if paths["pitch_C"]:
        frames.append(_load_pitch_csv(paths["pitch_C"], "C", method_filter="carnatic"))

    missing = [label for label, p in paths.items() if p is None]
    if missing:
        msg = f"Missing result files for: {', '.join(missing)}"
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg)

    if not frames:
        raise RuntimeError(
            "No result files found under "
            f"{results_root}. Run the §3 and §4 experiments first."
        )

    return pd.concat(frames, ignore_index=True)


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a long-form result table into a 3 × 2 summary per (task, metric).

    Output columns: task, metric, condition, domain, mean, std, n.
    """
    grouped = (
        long_df
        .groupby(["task", "metric", "condition", "domain"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return grouped.round(4)


def pivot_table(summary: pd.DataFrame, metric: str, task: str) -> pd.DataFrame:
    """Convenience: pivot the long summary into a printable 3 × 2 table for ONE metric."""
    df = summary[(summary["task"] == task) & (summary["metric"] == metric)]
    if df.empty:
        return pd.DataFrame()
    return (
        df.pivot(index="condition", columns="domain", values="mean")
          .reindex(index=["A", "B", "C"], columns=["western", "carnatic"])
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# ── Backwards-compat shims ────────────────────────────────────────────────────
# The old API was: run_beat_cross_domain(carnatic_tracks, western_tracks).
# Keep those names importable in case any notebook still uses them, but they
# now point at the aggregator path rather than re-running models.

def run_beat_cross_domain(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "run_beat_cross_domain() is deprecated. The cross-domain step now "
        "aggregates teammates' existing result files; use load_all_results() "
        "and summarise() instead, or call `python scripts/run_cross_domain.py`."
    )


def run_pitch_cross_domain(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "run_pitch_cross_domain() is deprecated. See run_beat_cross_domain()."
    )


def summarise_cross_domain(df: pd.DataFrame, task: str | None = None) -> pd.DataFrame:
    """Old name kept for compatibility with notebooks."""
    s = summarise(df)
    if task is not None:
        s = s[s["task"] == task]
    return s
