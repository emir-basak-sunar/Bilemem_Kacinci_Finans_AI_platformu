"""
FinAI Training Configuration — v2.0
=====================================
Central config for all training notebooks and serving pipeline.

Değişiklikler v2.0 (hedge-fund grade):
- CRYPTO_UNIVERSE ayrıldı — kripto hisselerle aynı modelde eğitilmez
- EQUITY_UNIVERSE: sadece hisse senetleri
- OptunaConfig: SQLite persistence — tuning geçmişi kaybolmaz
- XGBoostConfig: min_child_weight, gamma eklendi (kritik regularization)
- LGBMConfig: min_child_samples, min_split_gain eklendi
- CrossSectionalConfig: rank-based IC optimization için
- RegimeConfig: bull/bear/sideways rejim tespiti için
- ARIMAConfig: m=5 (haftalık) — m=252 Colab'da saatler sürer, anlamsız
- LSTMConfig/TCNConfig/TFTConfig: A100 için optimize edildi (daha büyük modeller)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

# ============================================================
# Symbols & Universe
# ============================================================

# Core watchlist — trained with per-symbol fine-tuning
CORE_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "IBM",
]

# Crypto (Yahoo Finance format) — AYRI MODEL, hisselerle karıştırılmaz
# Kripto: 7/24 işlem, vol 5-10x hisse, farklı dinamikler
CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD",
]

# Equity universe — sadece hisse senetleri (kripto hariç)
EQUITY_UNIVERSE = CORE_SYMBOLS + [
    "META", "NFLX", "AMD", "INTC", "JPM", "GS", "BAC",
    "V", "MA", "DIS", "PYPL", "SQ", "COIN", "UBER",
    "CRM", "ADBE", "ORCL", "CSCO", "QCOM", "AVGO",
]

# Extended universe for universal model training (equity only)
TRAINING_UNIVERSE = EQUITY_UNIVERSE

# All symbols (equity + crypto) — veri çekme için
ALL_SYMBOLS = EQUITY_UNIVERSE + CRYPTO_SYMBOLS

# FIX v1.1: Sabit sembol encoding — her çalıştırmada aynı mapping
SYMBOL_ENCODING: Dict[str, int] = {sym: i for i, sym in enumerate(sorted(ALL_SYMBOLS))}

# Equity-only encoding (kripto hariç)
EQUITY_ENCODING: Dict[str, int] = {sym: i for i, sym in enumerate(sorted(EQUITY_UNIVERSE))}


# ============================================================
# Feature Engineering Config
# ============================================================

@dataclass
class FeatureConfig:
    # Trend indicators
    sma_windows: List[int] = field(default_factory=lambda: [7, 14, 21, 50, 100, 200])
    ema_windows: List[int] = field(default_factory=lambda: [7, 14, 21, 50, 100, 200])

    # Momentum
    rsi_window: int = 14
    stoch_window: int = 14
    stoch_smooth: int = 3
    cci_window: int = 20
    roc_windows: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Volatility
    bb_window: int = 20
    bb_std: int = 2
    atr_window: int = 14

    # Volume
    mfi_window: int = 14

    # Lag features
    price_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    volume_lags: List[int] = field(default_factory=lambda: [1, 2, 3])
    return_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])

    # LSTM / TCN / TFT sequence length — A100'de 120 bar rahat çalışır
    sequence_length: int = 120

    # Prediction horizons (multi-step)
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Cross-sectional ranking window (kaç sembol arasında rank)
    cs_rank_min_symbols: int = 5  # En az bu kadar sembol olmalı


# ============================================================
# Model Config — A100 için optimize edildi
# ============================================================

@dataclass
class XGBoostConfig:
    n_estimators: int = 1000       # v2.0: 500 → 1000 (daha fazla ağaç, düşük LR ile)
    learning_rate: float = 0.01    # v2.0: 0.03 → 0.01 (daha yavaş, daha iyi)
    max_depth: int = 6             # v2.0: 8 → 6 (overfitting azalt)
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 20     # v2.0: YENİ — finansal veri için kritik
    gamma: float = 0.1             # v2.0: YENİ — split için min gain
    n_jobs: int = -1
    early_stopping_rounds: int = 50  # v2.0: 20 → 50 (daha sabırlı)


@dataclass
class LGBMConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.01
    num_leaves: int = 31           # v2.0: 63 → 31 (overfitting azalt)
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_samples: int = 20    # v2.0: YENİ — minimum leaf samples
    min_split_gain: float = 0.01   # v2.0: YENİ — split için min gain
    n_jobs: int = -1
    early_stopping_rounds: int = 50


@dataclass
class CatBoostConfig:
    iterations: int = 1000
    learning_rate: float = 0.01
    depth: int = 6
    l2_leaf_reg: float = 3.0
    min_data_in_leaf: int = 20     # v2.0: YENİ
    early_stopping_rounds: int = 50


@dataclass
class LSTMConfig:
    hidden_size: int = 256         # v2.0: 128 → 256 (A100'de rahat)
    num_layers: int = 3            # v2.0: 2 → 3
    dropout: float = 0.2
    bidirectional: bool = True
    batch_size: int = 256          # v2.0: 64 → 256 (A100 için)
    epochs: int = 100              # v2.0: 50 → 100
    lr: float = 0.0005             # v2.0: 0.001 → 0.0005
    patience: int = 20             # v2.0: 10 → 20


@dataclass
class TCNConfig:
    # Derin receptive field: kernel_size=3, 8 katman → 2*(3-1)*(1+2+4+8+16+32+64+128)=1020 bar
    num_channels: List[int] = field(
        default_factory=lambda: [64, 64, 128, 128, 256, 256, 128, 64]
    )
    kernel_size: int = 3
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 150
    lr: float = 0.0003
    patience: int = 20


@dataclass
class TFTConfig:
    hidden_size: int = 128         # v2.0: 64 → 128
    attention_head_size: int = 8   # v2.0: 4 → 8
    dropout: float = 0.1
    hidden_continuous_size: int = 64  # v2.0: 32 → 64
    batch_size: int = 128          # v2.0: 32 → 128
    epochs: int = 150
    lr: float = 0.0002
    patience: int = 20
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


@dataclass
class ARIMAConfig:
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    seasonal: bool = False         # v2.0: True → False — log-return'de seasonality yok
    # v2.0: m=252 → m=5 — günlük log-return'de yıllık seasonality anlamsız
    # m=252 ile SARIMA Colab'da saatler sürer ve IC≈0 üretir
    m: int = 5
    stepwise: bool = True
    suppress_warnings: bool = True


# ============================================================
# v2.0: Yeni Config'ler
# ============================================================

@dataclass
class OptunaConfig:
    """Optuna hyperparameter search configuration."""
    n_trials_universal: int = 50
    n_trials_sector: int = 20
    n_trials_crypto: int = 30
    # SQLite persistence — Colab session kapanınca tuning geçmişi kaybolmaz
    # optuna>=3.5 ile SQLite built-in gelir, ayrı paket gerekmez
    # Not: Gerçek yol aşağıda OPTUNA_CFG oluşturulduktan sonra set edilir
    storage: str = ""
    # Arama alanı genişletildi
    xgb_n_estimators_range: tuple = (200, 2000)
    xgb_lr_range: tuple = (0.005, 0.1)
    xgb_depth_range: tuple = (3, 8)
    xgb_min_child_weight_range: tuple = (5, 50)
    lgbm_num_leaves_range: tuple = (15, 63)
    lgbm_min_child_samples_range: tuple = (10, 100)


@dataclass
class CrossSectionalConfig:
    """Cross-sectional (rank-based) alpha framework config."""
    # Minimum sembol sayısı — cross-sectional rank için
    min_symbols: int = 5
    # Long-short portfolio: top/bottom N sembol
    long_short_n: int = 3
    # Rank normalization: True = [-0.5, 0.5], False = [0, 1]
    center_ranks: bool = True
    # IC hesaplama penceresi (walk-forward fold başına)
    ic_window: int = 63


@dataclass
class RegimeConfig:
    """Market regime detection config."""
    # VIX eşikleri
    vix_low: float = 15.0      # Düşük volatilite (bull)
    vix_high: float = 25.0     # Yüksek volatilite (bear/stress)
    vix_extreme: float = 40.0  # Kriz
    # S&P500 trend
    sp500_ma_window: int = 200
    # Rejim geçiş smoothing (ani geçişleri önle)
    regime_smooth_window: int = 5
    # Ensemble ağırlıkları per rejim
    # Format: {regime: {model: weight}}
    ensemble_weights_by_regime: dict = field(default_factory=lambda: {
        "bull_low_vol":   {"xgboost": 0.3, "lightgbm": 0.3, "catboost": 0.2, "lstm": 0.2},
        "bear_high_vol":  {"lstm": 0.4, "tft": 0.3, "xgboost": 0.2, "lightgbm": 0.1},
        "sideways":       {"tft": 0.4, "lstm": 0.3, "xgboost": 0.2, "lightgbm": 0.1},
        "crisis":         {"lstm": 0.5, "tft": 0.3, "xgboost": 0.2},
    })


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_REGISTRY_DIR = os.path.join(TRAINING_DIR, "model_registry")
MODELS_DIR = os.path.join(MODEL_REGISTRY_DIR, "models")
MANIFESTS_DIR = os.path.join(MODEL_REGISTRY_DIR, "manifests")
METRICS_DIR = os.path.join(MODEL_REGISTRY_DIR, "metrics")
DATA_DIR = os.path.join(TRAINING_DIR, "data")

# Ensure directories exist
for d in [MODELS_DIR, MANIFESTS_DIR, METRICS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Sub-directories per model type
for model_type in [
    "xgboost", "lightgbm", "catboost",
    "lstm", "tcn", "tft",
    "arima", "sarima", "sarimax",
    "ensemble", "crypto_xgboost", "crypto_lstm",
]:
    os.makedirs(os.path.join(MODELS_DIR, model_type), exist_ok=True)


# ============================================================
# Default instances
# ============================================================

FEATURE_CFG    = FeatureConfig()
XGBOOST_CFG    = XGBoostConfig()
LGBM_CFG       = LGBMConfig()
CATBOOST_CFG   = CatBoostConfig()
LSTM_CFG       = LSTMConfig()
TCN_CFG        = TCNConfig()
TFT_CFG        = TFTConfig()
ARIMA_CFG      = ARIMAConfig()
OPTUNA_CFG     = OptunaConfig()
OPTUNA_CFG.storage = f"sqlite:///{os.path.join(TRAINING_DIR, 'optuna_studies.db')}"
CS_CFG         = CrossSectionalConfig()
REGIME_CFG     = RegimeConfig()
