"""Cross-domain experiments: train on one domain, evaluate on the other."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing.common import AudioTrack
from src.rhythm.beat_tracking import run_beat_tracking
from src.rhythm.evaluation import evaluate_beats, aggregate_results as agg_beats
from src.pitch.estimation import run_pitch_estimation
from src.pitch.evaluation import evaluate_pitch, aggregate_results as agg_pitch


def run_beat_cross_domain(
    carnatic_tracks: list[AudioTrack],
    western_tracks: list[AudioTrack],
    backend: str = "madmom",
) -> pd.DataFrame:
    """Evaluate beat tracker (trained on Western data via madmom defaults) on both domains."""
    rows = []
    for tracks in [carnatic_tracks, western_tracks]:
        for track in tracks:
            pred = run_beat_tracking(track, backend=backend)
            result = evaluate_beats(track, pred)
            rows.append(result.__dict__)
    return pd.DataFrame(rows)


def run_pitch_cross_domain(
    carnatic_tracks: list[AudioTrack],
    western_tracks: list[AudioTrack],
    method: str = "crepe",
    **kwargs,
) -> pd.DataFrame:
    """Evaluate pitch estimator on both domains."""
    rows = []
    for tracks in [carnatic_tracks, western_tracks]:
        for track in tracks:
            pred = run_pitch_estimation(track, method=method, **kwargs)
            result = evaluate_pitch(track, pred)
            rows.append(result.__dict__)
    return pd.DataFrame(rows)


def summarise_cross_domain(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Group by domain and compute mean ± std for all numeric columns."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    summary = (
        df.groupby("domain")[numeric]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary["task"] = task
    return summary


def save_results(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
