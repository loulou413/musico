"""Pitch evaluation metrics using mir_eval and cents-based analysis."""

import numpy as np
import mir_eval
from dataclasses import dataclass

from src.preprocessing.common import AudioTrack, hz_to_cents, snap_to_semitone
from .estimation import PitchPrediction


@dataclass
class PitchEvalResult:
    track_id: str
    domain: str
    method: str
    # mir_eval melody metrics
    voicing_recall: float
    voicing_false_alarm: float
    raw_pitch_accuracy: float       # RPA: correct pitch ignoring voicing
    raw_chroma_accuracy: float      # RCA: octave-invariant
    overall_accuracy: float         # OA: correct pitch + voicing
    # Cents-level analysis
    mean_abs_error_cents: float     # mean |error| in cents over voiced frames
    median_abs_error_cents: float
    # Semitone snapping bias (how much CREPE is pulled to ET grid)
    semitone_snap_ratio: float      # fraction of errors < 5 cents (near a semitone)


def evaluate_pitch(
    track: AudioTrack,
    prediction: PitchPrediction,
    cents_tolerance: float = 50.0,
) -> PitchEvalResult:
    if track.f0_times is None or track.f0_freqs is None:
        return _empty_result(track, prediction)

    ref_times = track.f0_times
    ref_freqs = track.f0_freqs
    est_times = prediction.times
    est_freqs = prediction.frequencies

    # Resample estimate to reference time grid
    est_freqs_aligned = _align_to_ref(ref_times, est_times, est_freqs)

    ref_voiced = ref_freqs > 0
    est_voiced = est_freqs_aligned > 0

    # mir_eval melody evaluation
    scores = mir_eval.melody.evaluate(
        ref_time=ref_times,
        ref_freq=ref_freqs,
        est_time=ref_times,
        est_freq=est_freqs_aligned,
    )

    # Cents error (voiced frames only)
    voiced_mask = ref_voiced & est_voiced
    cents_error = _cents_error_voiced(
        ref_freqs[voiced_mask], est_freqs_aligned[voiced_mask]
    )
    mae = float(np.mean(np.abs(cents_error))) if len(cents_error) > 0 else float("nan")
    medae = float(np.median(np.abs(cents_error))) if len(cents_error) > 0 else float("nan")

    # Semitone snapping: compare est to its nearest ET semitone
    snap_ratio = _semitone_snap_ratio(est_freqs_aligned[voiced_mask]) if voiced_mask.any() else float("nan")

    return PitchEvalResult(
        track_id=track.track_id,
        domain=track.domain,
        method=prediction.method,
        voicing_recall=scores["Voicing Recall"],
        voicing_false_alarm=scores["Voicing False Alarm"],
        raw_pitch_accuracy=scores["Raw Pitch Accuracy"],
        raw_chroma_accuracy=scores["Raw Chroma Accuracy"],
        overall_accuracy=scores["Overall Accuracy"],
        mean_abs_error_cents=mae,
        median_abs_error_cents=medae,
        semitone_snap_ratio=snap_ratio,
    )


def aggregate_results(results: list[PitchEvalResult]) -> dict:
    metrics = [
        "voicing_recall", "voicing_false_alarm", "raw_pitch_accuracy",
        "raw_chroma_accuracy", "overall_accuracy",
        "mean_abs_error_cents", "median_abs_error_cents", "semitone_snap_ratio",
    ]
    agg = {}
    for m in metrics:
        vals = np.array([getattr(r, m) for r in results if not np.isnan(getattr(r, m))])
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return agg


# ── helpers ──────────────────────────────────────────────────────────────────

def _align_to_ref(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    est_freqs: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour interpolation of est_freqs onto ref_times grid."""
    indices = np.searchsorted(est_times, ref_times)
    indices = np.clip(indices, 0, len(est_freqs) - 1)
    return est_freqs[indices]


def _cents_error_voiced(
    ref_freqs: np.ndarray,
    est_freqs: np.ndarray,
) -> np.ndarray:
    ref_cents = hz_to_cents(ref_freqs)
    est_cents = hz_to_cents(est_freqs)
    return est_cents - ref_cents


def _semitone_snap_ratio(freqs_hz: np.ndarray, threshold_cents: float = 5.0) -> float:
    """Fraction of frames where the estimate is within threshold_cents of an ET note."""
    if len(freqs_hz) == 0:
        return float("nan")
    snapped = snap_to_semitone(freqs_hz)
    voiced = freqs_hz > 0
    ref_cents = hz_to_cents(freqs_hz[voiced])
    snap_cents = hz_to_cents(snapped[voiced])
    near_semitone = np.abs(ref_cents - snap_cents) < threshold_cents
    return float(np.mean(near_semitone))


def _empty_result(track: AudioTrack, pred: PitchPrediction) -> PitchEvalResult:
    nan = float("nan")
    return PitchEvalResult(
        track_id=track.track_id,
        domain=track.domain,
        method=pred.method,
        voicing_recall=nan,
        voicing_false_alarm=nan,
        raw_pitch_accuracy=nan,
        raw_chroma_accuracy=nan,
        overall_accuracy=nan,
        mean_abs_error_cents=nan,
        median_abs_error_cents=nan,
        semitone_snap_ratio=nan,
    )
