"""Beat activation model: CNN front-end + BiLSTM temporal model.

Architecture mirrors the classic Böck et al. RNN beat tracker but is
lightweight enough to train on a single GPU (or CPU for small datasets).

Input:  (batch, n_mels, seq_len)   — log-mel spectrogram
Output: (batch, seq_len)           — beat activation in [0, 1]

Post-processing (DBN from madmom or simple peak picking) converts the
activation to discrete beat times at inference time.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeatActivationModel(nn.Module):
    """CNN + BiLSTM beat activation predictor.

    Args:
        n_mels:      Number of mel frequency bands (input height).
        cnn_channels: List of channel sizes for the CNN front-end.
        lstm_hidden: Hidden size per direction of the BiLSTM.
        lstm_layers: Number of BiLSTM layers.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        n_mels: int = 128,
        cnn_channels: list[int] = [32, 64],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        cnn_in = n_mels
        cnn_blocks = []
        for out_ch in cnn_channels:
            cnn_blocks.append(ConvBlock(cnn_in, out_ch, kernel=3, dropout=dropout))
            cnn_in = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)                   # (B, C, T)
        feat = feat.permute(0, 2, 1)         # (B, T, C)
        lstm_out, _ = self.lstm(feat)        # (B, T, 2*lstm_hidden)
        return self.head(lstm_out).squeeze(-1)


def activation_to_beat_times(
    activation: torch.Tensor,
    sr: int = 22050,
    hop_length: int = 512,
    threshold: float = 0.3,
    min_gap_frames: int = 10,
) -> "np.ndarray":
    """Simple peak-picking on the beat activation to get beat times in seconds.

    For higher accuracy, replace this with madmom's DBN post-processor.
    """
    import numpy as np
    from scipy.signal import find_peaks

    act_np = activation.squeeze().cpu().numpy()
    peaks, _ = find_peaks(act_np, height=threshold, distance=min_gap_frames)
    times = peaks * hop_length / sr
    return times
