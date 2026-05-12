"""Rhythm evaluation metrics using mir_eval."""

import numpy as np
import mir_eval
from dataclasses import dataclass

from src.preprocessing.common import AudioTrack
from .beat_tracking import BeatPrediction


@dataclass
class BeatEvalResult:
    track_id: str
    domain: str
    f_measure: float     # mir_eval beat F-measure (70 ms window)
    cemgil: float        # Cemgil score (Gaussian window)
    continuity: float    # CMLt — correct metrical level total
    information_gain: float
    # Downbeat metrics
    downbeat_f_measure: float
    # Extra diagnostics
    n_ref_beats: int
    n_est_beats: int
    tempo_ref_bpm: float
    tempo_est_bpm: float


def evaluate_beats(
    track: AudioTrack,
    prediction: BeatPrediction,
    tolerance: float = 0.07,
) -> BeatEvalResult:
    ref_beats = track.beat_times if track.beat_times is not None else np.array([])
    est_beats = prediction.beat_times

    # mir_eval requires sorted, non-empty arrays
    ref_beats = np.sort(ref_beats)
    est_beats = np.sort(est_beats)

    if len(ref_beats) == 0 or len(est_beats) == 0:
        return _empty_result(track, prediction)

    scores = mir_eval.beat.evaluate(ref_beats, est_beats)

    # Downbeat evaluation
    ref_db = track.downbeat_times if track.downbeat_times is not None else np.array([])
    est_db = prediction.downbeat_times
    if len(ref_db) > 0 and len(est_db) > 0:
        db_scores = mir_eval.beat.evaluate(np.sort(ref_db), np.sort(est_db))
        db_f = db_scores["F-measure"]
    else:
        db_f = float("nan")

    tempo_ref = 60.0 / np.median(np.diff(ref_beats)) if len(ref_beats) > 1 else 0.0

    return BeatEvalResult(
        track_id=track.track_id,
        domain=track.domain,
        f_measure=scores["F-measure"],
        cemgil=scores["Cemgil"],
        continuity=scores["CMLt"],
        information_gain=scores["Information gain"],
        downbeat_f_measure=db_f,
        n_ref_beats=len(ref_beats),
        n_est_beats=len(est_beats),
        tempo_ref_bpm=tempo_ref,
        tempo_est_bpm=prediction.bpm,
    )


def aggregate_results(results: list[BeatEvalResult]) -> dict:
    """Compute mean ± std for each metric across a list of results."""
    metrics = ["f_measure", "cemgil", "continuity", "information_gain", "downbeat_f_measure"]
    agg = {}
    for m in metrics:
        vals = np.array([getattr(r, m) for r in results if not np.isnan(getattr(r, m))])
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return agg


def _empty_result(track: AudioTrack, pred: BeatPrediction) -> BeatEvalResult:
    return BeatEvalResult(
        track_id=track.track_id,
        domain=track.domain,
        f_measure=0.0,
        cemgil=0.0,
        continuity=0.0,
        information_gain=0.0,
        downbeat_f_measure=float("nan"),
        n_ref_beats=0,
        n_est_beats=len(pred.beat_times),
        tempo_ref_bpm=0.0,
        tempo_est_bpm=pred.bpm,
    )
