"""Pitch error analysis: gamaka detection, cents distributions, snapping bias."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.preprocessing.common import hz_to_cents, snap_to_semitone
from .evaluation import PitchEvalResult


def results_to_dataframe(results: list[PitchEvalResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def plot_metric_by_domain(df: pd.DataFrame, metric: str, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="domain", y=metric, hue="method", ax=ax)
    ax.set_title(f"Pitch — {metric}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / f"pitch_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cents_error_histogram(
    ref_freqs: np.ndarray,
    est_freqs: np.ndarray,
    domain: str,
    output_dir: str,
    bins: int = 120,
) -> None:
    """Histogram of pitch errors in cents, with semitone grid overlay."""
    voiced = (ref_freqs > 0) & (est_freqs > 0)
    if not voiced.any():
        return
    errors = hz_to_cents(est_freqs[voiced]) - hz_to_cents(ref_freqs[voiced])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errors, bins=bins, density=True, color="steelblue", alpha=0.7)
    # Mark semitone boundaries (every 100 cents)
    for s in range(-6, 7):
        ax.axvline(s * 100, color="red", lw=0.5, alpha=0.5)
    ax.set_xlabel("Pitch error (cents)")
    ax.set_ylabel("Density")
    ax.set_title(f"Pitch error distribution — {domain}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        Path(output_dir) / f"cents_hist_{domain}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def detect_gamaka_regions(
    f0_times: np.ndarray,
    f0_freqs: np.ndarray,
    smoothing_window: int = 5,
    modulation_threshold_cents: float = 30.0,
) -> np.ndarray:
    """Return boolean mask of frames likely containing gamakas.

    Gamakas are characterised by rapid, continuous pitch modulation > threshold.
    """
    if len(f0_freqs) == 0:
        return np.array([], dtype=bool)

    voiced = f0_freqs > 0
    cents = np.where(voiced, hz_to_cents(np.where(voiced, f0_freqs, 1.0)), np.nan)

    # Local range in a sliding window
    half = smoothing_window // 2
    n = len(cents)
    local_range = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = cents[lo:hi]
        window = window[~np.isnan(window)]
        if len(window) > 1:
            local_range[i] = np.max(window) - np.min(window)

    return local_range > modulation_threshold_cents


def plot_pitch_contour(
    f0_times: np.ndarray,
    f0_ref: np.ndarray,
    f0_est: np.ndarray,
    gamaka_mask: np.ndarray | None,
    track_id: str,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    voiced_ref = f0_ref > 0
    voiced_est = f0_est > 0
    ax.plot(f0_times[voiced_ref], hz_to_cents(f0_ref[voiced_ref]), lw=1, label="Reference", color="black")
    ax.plot(f0_times[voiced_est], hz_to_cents(f0_est[voiced_est]), lw=1, label="Estimated", color="steelblue", alpha=0.8)
    if gamaka_mask is not None:
        gamaka_times = f0_times[gamaka_mask & voiced_ref]
        if len(gamaka_times) > 0:
            ax.scatter(
                gamaka_times,
                hz_to_cents(f0_ref[gamaka_mask & voiced_ref]),
                c="red", s=4, zorder=5, label="Gamaka region",
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (cents re. A=440)")
    ax.set_title(f"Pitch contour — {track_id}")
    ax.legend(fontsize=8)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(
        Path(output_dir) / f"contour_{track_id.replace('/', '_')}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def gamaka_error_vs_non_gamaka(
    ref_freqs: np.ndarray,
    est_freqs: np.ndarray,
    gamaka_mask: np.ndarray,
) -> dict:
    """Compare mean absolute cents error inside vs outside gamaka regions."""
    voiced = (ref_freqs > 0) & (est_freqs > 0)
    errors_cents = np.abs(hz_to_cents(est_freqs[voiced]) - hz_to_cents(ref_freqs[voiced]))
    mask_v = gamaka_mask[voiced]
    return {
        "mae_gamaka": float(np.mean(errors_cents[mask_v])) if mask_v.any() else float("nan"),
        "mae_non_gamaka": float(np.mean(errors_cents[~mask_v])) if (~mask_v).any() else float("nan"),
    }
