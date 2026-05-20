"""CREPE pitch estimation CNN (PyTorch).

Architecture is a faithful port of the official CREPE 'full' model
(Kim et al., 2018), so the official pretrained Keras weights can be
loaded into it (see scripts/convert_crepe_weights.py).

Layer spec (matches crepe.core.build_and_load_model('full')):
  conv1: 1024 filters, kernel (512, 1), stride (4, 1), 'same' padding, ReLU
  conv2-4: 128 filters, kernel (64, 1), stride 1, 'same' padding, ReLU
  conv5: 256 filters, kernel (64, 1), stride 1, 'same' padding, ReLU
  conv6: 512 filters, kernel (64, 1), stride 1, 'same' padding, ReLU
Each conv is followed by BatchNorm → MaxPool(2, 1, 'valid') → Dropout.
Final: Flatten → Dense(360, sigmoid).

Total params: ~22.2M (matches Keras model count).

Input:  (batch, 1, 1024)   — normalised 64 ms audio frame at 16 kHz
Output: (batch, 360)        — per-bin activation in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dataset import PITCH_BINS_HZ, PITCH_BINS_CENTS, N_BINS


_LAYERS = [
    # (n_filters, kernel, stride)
    (1024, 512, 4),
    (128,   64, 1),
    (128,   64, 1),
    (128,   64, 1),
    (256,   64, 1),
    (512,   64, 1),
]


def _tf_same_pad_1d(in_len: int, kernel: int, stride: int) -> tuple[int, int]:
    """Compute asymmetric TF 'same' padding for stride > 1.

    out = ceil(in / stride), total_pad = max((out - 1) * stride + kernel - in, 0).
    TF puts the extra pixel on the right.
    """
    out_len = (in_len + stride - 1) // stride
    total = max((out_len - 1) * stride + kernel - in_len, 0)
    left = total // 2
    right = total - left
    return left, right


class _CREPEConvBlock(nn.Module):
    """conv (with TF 'same' padding) → BN → ReLU → MaxPool(2,'valid') → Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        in_len: int,
        dropout: float = 0.25,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.01,   # PyTorch momentum = 1 - TF momentum (0.99)
    ):
        super().__init__()
        self.pad_left, self.pad_right = _tf_same_pad_1d(in_len, kernel, stride)
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel, 1),
            stride=(stride, 1),
            padding=0,   # we pad manually
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))   # 'valid'
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L, 1)
        # Order matches Keras CREPE: Conv(relu) → BN → MaxPool → Dropout.
        # (Keras Conv2D bakes the activation into the conv layer itself, so ReLU
        # is applied *before* BN here.)
        x = F.pad(x, (0, 0, self.pad_left, self.pad_right))   # pad along L only
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CREPELike(nn.Module):
    """Faithful PyTorch port of CREPE 'full'.

    The signature keeps `n_filters_scale` for backward compatibility with
    existing checkpoints/CLI flags. Only `n_filters_scale=1.0` is compatible
    with the official CREPE weights — other values yield a from-scratch model
    that won't accept the converted weights.
    """

    def __init__(self, n_filters_scale: float = 1.0, dropout: float = 0.25):
        super().__init__()
        self.n_filters_scale = n_filters_scale

        cur_len = 1024
        in_ch = 1
        blocks = []
        for (n_filt, kernel, stride) in _LAYERS:
            out_ch = max(1, int(round(n_filt * n_filters_scale)))
            blocks.append(_CREPEConvBlock(
                in_ch=in_ch, out_ch=out_ch,
                kernel=kernel, stride=stride,
                in_len=cur_len, dropout=dropout,
            ))
            cur_len = (cur_len + stride - 1) // stride
            cur_len = cur_len // 2
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.flatten_size = in_ch * cur_len
        self.classifier = nn.Linear(self.flatten_size, N_BINS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 1024) → (B, 360)"""
        x = x.unsqueeze(-1)
        for blk in self.blocks:
            x = blk(x)
        # Emulate Keras Permute((3,1,2)) + Flatten: transpose to (B,H,W,C) before flattening
        # so weight layout matches the converted CREPE checkpoint.
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


def activation_to_hz(activation: np.ndarray) -> float:
    """Weighted mean of pitch bins — smoother than argmax for gamakas."""
    return float(np.sum(PITCH_BINS_HZ * activation) / (activation.sum() + 1e-8))


def activation_to_hz_viterbi(activations: np.ndarray, transition_sigma: float = 1.0) -> np.ndarray:
    """Viterbi decoding over a sequence of activations → Hz per frame."""
    T = len(activations)
    log_emit = np.log(activations + 1e-8)
    bin_dist = np.abs(np.arange(N_BINS)[:, None] - np.arange(N_BINS)[None, :])
    log_trans = -0.5 * (bin_dist / transition_sigma) ** 2

    viterbi = np.full((T, N_BINS), -np.inf)
    backptr = np.zeros((T, N_BINS), dtype=int)
    viterbi[0] = log_emit[0]
    for t in range(1, T):
        scores = viterbi[t - 1][:, None] + log_trans
        best_prev = np.argmax(scores, axis=0)
        viterbi[t] = scores[best_prev, np.arange(N_BINS)] + log_emit[t]
        backptr[t] = best_prev
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(viterbi[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return PITCH_BINS_HZ[path]
