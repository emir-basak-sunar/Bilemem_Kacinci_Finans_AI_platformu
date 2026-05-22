"""
FinAI DL Architectures — BiLSTM, TCN, TFT, GRN
================================================
Tüm DL model mimarileri bu dosyada tanımlanır.
train_lstm.py ve train_tcn_tft.py bu dosyadan import eder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ============================================================
# 1. BIDIRECTIONAL LSTM
# ============================================================

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM with LayerNorm + GELU.
    A100 için: hidden_size=256, num_layers=3
    """
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 3, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        direction_mult = 2 if bidirectional else 1
        lstm_out_size = hidden_size * direction_mult
        self.layer_norm = nn.LayerNorm(lstm_out_size)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        return self.fc(last_hidden).squeeze(-1)


# ============================================================
# 2. TEMPORAL CNN (TCN)
# ============================================================

class TCNBlock(nn.Module):
    """
    TCN residual block with CAUSAL dilated convolution.
    F.pad ile sadece sol tarafa padding → causal convolution.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.dropout  = nn.Dropout(dropout)
        self.relu     = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = F.pad(x, (self.causal_pad, 0))
        out = self.dropout(self.relu(self.bn1(self.conv1(out))))
        out = F.pad(out, (self.causal_pad, 0))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network.
    8 katman, receptive field ~1020 bar (A100 için).
    """
    def __init__(self, input_size: int, num_channels: List[int] = None,
                 kernel_size: int = 3, dropout: float = 0.15):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128, 256, 256, 128, 64]

        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, seq, feat) → (batch, feat, seq)
        out = self.network(x)
        out = out[:, :, -1]       # son zaman adımı
        return self.fc(out).squeeze(-1)


# ============================================================
# 3. GATED RESIDUAL NETWORK
# ============================================================

class GatedResidualNetwork(nn.Module):
    """
    GRN — orijinal TFT paper'dan.
    GLU (Gated Linear Unit) ile önemli feature'ları seçer.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.residual_proj = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        h, gate = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)
        return self.layer_norm(h + residual)


# ============================================================
# 4. TFT (Temporal Fusion Transformer)
# ============================================================

class TFTModel(nn.Module):
    """
    Simplified TFT:
    - Variable Selection Network
    - Gated Residual Network (GRN)
    - Multi-head attention
    - Quantile outputs (7 quantile: P5, P10, P25, P50, P75, P90, P95)
    """
    def __init__(self, input_size: int, hidden_size: int = 128,
                 n_heads: int = 8, dropout: float = 0.1,
                 quantiles: List[float] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.quantiles = quantiles

        self.vsn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1),
        )
        self.grn = GatedResidualNetwork(input_size, hidden_size, dropout)
        self.encoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, dropout=dropout, batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in quantiles
        ])

    def forward(self, x):
        weights = self.vsn(x)
        x_selected = x * weights
        x_transformed = self.grn(x_selected)
        encoded, _ = self.encoder(x_transformed)
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = self.layer_norm1(attended + encoded)
        attended = self.post_attn_grn(attended)
        attended = self.layer_norm2(attended)
        last = attended[:, -1, :]
        return torch.stack([head(last).squeeze(-1) for head in self.quantile_heads], dim=-1)

    def predict_point(self, x):
        """Median (P50) prediction."""
        preds = self.forward(x)
        return preds[:, self.quantiles.index(0.5)]
