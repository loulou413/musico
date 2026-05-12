"""Error analysis for beat tracking results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .evaluation import BeatEvalResult


def results_to_dataframe(results: list[BeatEvalResult]) -> pd.DataFrame:
    rows = [r.__dict__ for r in results]
    return pd.DataFrame(rows)


def plot_metric_by_domain(df: pd.DataFrame, metric: str, output_dir: str) -> None:
    plot_df = df.dropna(subset=[metric])
    if plot_df.empty or plot_df["domain"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=plot_df, x="domain", y=metric, ax=ax)
    ax.set_title(f"Beat tracking — {metric}")
    ax.set_xlabel("Domain")
    ax.set_ylabel(metric)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / f"beat_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tempo_accuracy(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: estimated vs reference BPM, coloured by domain."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for domain, grp in df.groupby("domain"):
        ax.scatter(grp["tempo_ref_bpm"], grp["tempo_est_bpm"], label=domain, alpha=0.6)
    lim = (0, df[["tempo_ref_bpm", "tempo_est_bpm"]].max().max() * 1.05)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Reference BPM")
    ax.set_ylabel("Estimated BPM")
    ax.set_title("Tempo accuracy by domain")
    ax.legend()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / "tempo_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sam_alignment_analysis(
    ref_downbeats: np.ndarray,
    est_downbeats: np.ndarray,
    tolerance: float = 0.07,
) -> dict:
    """Compute how many predicted downbeats are within tolerance of a sam (downbeat)."""
    hits = 0
    for est in est_downbeats:
        if np.any(np.abs(ref_downbeats - est) <= tolerance):
            hits += 1
    precision = hits / len(est_downbeats) if len(est_downbeats) > 0 else 0.0
    recall = hits / len(ref_downbeats) if len(ref_downbeats) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "n_hits": hits}


def drift_analysis(
    ref_beats: np.ndarray,
    est_beats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ref_times, beat_error_seconds) for plotting drift over time."""
    errors = []
    ref_matched = []
    for est in est_beats:
        diffs = np.abs(ref_beats - est)
        idx = np.argmin(diffs)
        ref_matched.append(ref_beats[idx])
        errors.append(est - ref_beats[idx])
    return np.array(ref_matched), np.array(errors)


def plot_drift(
    ref_times: np.ndarray,
    errors: np.ndarray,
    track_id: str,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ref_times, errors * 1000, lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Reference beat time (s)")
    ax.set_ylabel("Error (ms)")
    ax.set_title(f"Beat drift — {track_id}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        Path(output_dir) / f"drift_{track_id.replace('/', '_')}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
