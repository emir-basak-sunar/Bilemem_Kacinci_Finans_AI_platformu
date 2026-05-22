"""
FinAI Training Pipeline — v2.0 (Modüler Yapı)
================================================
Google Colab A100 üzerinde çalıştırılmak üzere tasarlanmıştır.

Klasör yapısı:
    training/
    ├── features/               ← Feature engineering
    │   ├── feature_engineering.py  — 160+ teknik + cross-sectional feature
    │   ├── macro_features.py       — VIX, S&P500, yield curve, DXY, gold, oil
    │   └── feature_store.py        — Redis + Parquet feature store
    ├── models/
    │   ├── tree/               ← XGBoost / LightGBM / CatBoost
    │   │   └── train_tree_models.py
    │   ├── dl/                 ← Deep Learning
    │   │   ├── architectures.py    — BiLSTM, TCN, TFT, GRN
    │   │   ├── train_utils.py      — training loop, loss, ONNX export
    │   │   ├── train_lstm.py       — BiLSTM eğitimi
    │   │   └── train_tcn_tft.py    — TCN + TFT eğitimi
    │   └── statistical/        ← İstatistiksel modeller
    │       ├── train_arima.py      — ARIMA / SARIMA / SARIMAX
    │       └── train_garch.py      — GARCH(1,1) volatility
    ├── config.py               — Merkezi konfigürasyon
    ├── data_collection.py      — yfinance OHLCV + makro veri
    ├── walk_forward.py         — PurgedKFold, WalkForwardValidator, RegimeDetector
    ├── model_registry.py       — Model kaydetme, yükleme, versiyonlama
    └── run_all.py              — Tek komutla tüm pipeline

Çalıştırma:
    # Tüm pipeline
    python -m training.run_all

    # Seçici
    python -m training.run_all --only-tree
    python -m training.run_all --only-dl
    python -m training.run_all --only-statistical
    python -m training.run_all --skip-statistical

Colab kurulum:
    !pip install -r training/requirements_training.txt
"""

from training.config import (
    CORE_SYMBOLS,
    CRYPTO_SYMBOLS,
    EQUITY_UNIVERSE,
    ALL_SYMBOLS,
    SYMBOL_ENCODING,
    EQUITY_ENCODING,
    FEATURE_CFG,
    XGBOOST_CFG,
    LGBM_CFG,
    CATBOOST_CFG,
    LSTM_CFG,
    TCN_CFG,
    TFT_CFG,
    ARIMA_CFG,
    OPTUNA_CFG,
    CS_CFG,
    REGIME_CFG,
    DATA_DIR,
    MODEL_REGISTRY_DIR,
)

from training.model_registry import ModelRegistry, ModelManifest

# Feature engineering — doğrudan features subpackage'dan
try:
    from training.features.feature_engineering import (
        build_features,
        get_feature_columns,
        create_sequences,
        create_sequences_by_symbol,
        create_multi_horizon_targets,
        add_cross_sectional_features,
        add_cross_sectional_targets,
    )
except ImportError:
    pass

__version__ = "2.0.0"
__all__ = [
    "CORE_SYMBOLS", "CRYPTO_SYMBOLS", "EQUITY_UNIVERSE", "ALL_SYMBOLS",
    "SYMBOL_ENCODING", "EQUITY_ENCODING",
    "FEATURE_CFG", "XGBOOST_CFG", "LGBM_CFG", "CATBOOST_CFG",
    "LSTM_CFG", "TCN_CFG", "TFT_CFG", "ARIMA_CFG",
    "OPTUNA_CFG", "CS_CFG", "REGIME_CFG",
    "DATA_DIR", "MODEL_REGISTRY_DIR",
    "ModelRegistry", "ModelManifest",
]
