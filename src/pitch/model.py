"""CREPE-compatible pitch estimation CNN (PyTorch).

Architecture mirrors the original CREPE paper (Kim et al., 2018):
  6 × Conv1d blocks with increasing filter widths, batch norm, dropout
  → Flatten → Linear(360) → Sigmoid

Using the same 360-bin frequency grid as CREPE means:
  - Output probabilities are directly comparable to CREPE's output.
  - The trained Carnatic model can be evaluated with the same mir_eval pipeline.

Input:  (batch, 1, 1024)   — normalised 64 ms audio frame at 16 kHz
Output: (batch, 360)        — per-bin activation in [0, 1]
"""

import torch
import torch.nn as nn
import numpy as np

from .dataset import PITCH_BINS_HZ, PITCH_BINS_CENTS, N_BINS


# Layer specs from CREPE Table 1
_LAYERS = [
    # (n_filters, filter_width, stride)
    (1024, 512, 4),
    (128,   64, 1),
    (128,   64, 1),
    (128,   64, 1),
    (256,   64, 1),
    (512,   64, 1),
]


class CREPELike(nn.Module):
    """Lightweight CREPE-style CNN pitch estimator.

    The default capacity matches CREPE's "small" model; adjust n_filters_scale
    to shrink or grow all layers uniformly.
    """

    def __init__(self, n_filters_scale: float = 0.5, dropout: float = 0.25):
        super().__init__()
        layers = []
        in_ch = 1
        for n_filt, width, stride in _LAYERS:
            out_ch = max(1, int(n_filt * n_filters_scale))
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=width, stride=stride,
                          padding=width // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # Dynamically compute flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 1024)
            flat = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, N_BINS),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 1024) → (B, 360)"""
        feat = self.conv(x)
        return self.fc(feat.view(feat.size(0), -1))


# ── Inference helpers ─────────────────────────────────────────────────────────

def activation_to_hz(activation: np.ndarray) -> float:
    """Weighted mean of pitch bins — smoother than argmax for gamakas."""
    return float(np.sum(PITCH_BINS_HZ * activation) / (activation.sum() + 1e-8))


def activation_to_hz_viterbi(activations: np.ndarray, transition_sigma: float = 1.0) -> np.ndarray:
    """Viterbi decoding over a sequence of activations → Hz per frame.

    Penalises large pitch jumps between frames (important for gamakas vs noise).
    activations: (T, N_BINS)
    returns:     (T,) Hz array
    """
    T = len(activations)
    log_emit = np.log(activations + 1e-8)

    # Transition cost: Gaussian over bin distance
    bin_dist = np.abs(np.arange(N_BINS)[:, None] - np.arange(N_BINS)[None, :])
    log_trans = -0.5 * (bin_dist / transition_sigma) ** 2

    viterbi = np.full((T, N_BINS), -np.inf)
    backptr = np.zeros((T, N_BINS), dtype=int)
    viterbi[0] = log_emit[0]

    for t in range(1, T):
        scores = viterbi[t - 1][:, None] + log_trans   # (N_BINS, N_BINS)
        best_prev = np.argmax(scores, axis=0)            # (N_BINS,)
        viterbi[t] = scores[best_prev, np.arange(N_BINS)] + log_emit[t]
        backptr[t] = best_prev

    # Traceback
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(viterbi[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return PITCH_BINS_HZ[path]
