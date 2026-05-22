# FinAI Training — Colab A100 Kurulum Rehberi v2.0

## Klasör Yapısı

```
training/
├── features/                   ← Feature engineering
│   ├── feature_engineering.py  — 160+ teknik + cross-sectional feature
│   ├── macro_features.py       — VIX, S&P500, yield curve, DXY, gold, oil
│   └── feature_store.py        — Redis + Parquet feature store
├── models/
│   ├── tree/                   ← XGBoost / LightGBM / CatBoost
│   │   └── train_tree_models.py
│   ├── dl/                     ← Deep Learning (A100)
│   │   ├── architectures.py    — BiLSTM, TCN, TFT, GRN
│   │   ├── train_utils.py      — training loop, loss, ONNX export
│   │   ├── train_lstm.py       — BiLSTM eğitimi
│   │   └── train_tcn_tft.py    — TCN + TFT eğitimi
│   └── statistical/            ← İstatistiksel modeller
│       ├── train_arima.py      — ARIMA / SARIMA / SARIMAX
│       └── train_garch.py      — GARCH(1,1) volatility
├── config.py                   — Merkezi konfigürasyon
├── data_collection.py          — yfinance OHLCV + makro veri
├── walk_forward.py             — PurgedKFold, WalkForwardValidator, RegimeDetector
├── model_registry.py           — Model kaydetme, yükleme, versiyonlama
├── run_all.py                  — Tek komutla tüm pipeline
│
│ (Backward-compat shims — eski import'lar için)
├── feature_engineering.py      → training.features.feature_engineering
├── macro_features.py           → training.features.macro_features
└── feature_store.py            → training.features.feature_store
```

## Colab'a Yükleme

```python
# 1. Eski dosyaları temizle
import shutil, os
if os.path.exists('/content/training'):
    shutil.rmtree('/content/training')
print("Temizlendi")
```

Sonra tüm `training/` klasörünü Colab'a yükle (zip ile veya Drive mount).

## Kurulum

```bash
pip install -r training/requirements_training.txt
```

## Çalıştırma

```bash
# Tüm pipeline (statistical → tree → DL)
python -m training.run_all

# Seçici
python -m training.run_all --only-tree        # Sadece XGB/LGBM/CatBoost
python -m training.run_all --only-dl          # Sadece LSTM/TCN/TFT
python -m training.run_all --only-statistical # Sadece ARIMA/GARCH
python -m training.run_all --skip-statistical # Statistical hariç

# Tek modül
python -m training.models.tree.train_tree_models
python -m training.models.dl.train_lstm
python -m training.models.dl.train_tcn_tft
python -m training.models.statistical.train_arima
python -m training.models.statistical.train_garch
```

## Beklenen Çıktılar

```
# Tree Models
EQUITY_UNIVERSAL: IC=0.03-0.06, Sharpe=0.1-0.3
EQUITY_CS_RANK:   IC=0.03-0.05, Sharpe=1.0+ (cross-sectional)
EQUITY_H20:       Sharpe=1.8+, Calmar=3.7+
CRYPTO_UNIVERSAL: IC=0.05-0.08

# DL Models (A100)
LSTM: IC=0.02-0.04, Sharpe(LS)=0.07
TFT:  IC=0.02-0.04, CI_width=0.05

# Statistical
SARIMAX: IC=0.09-0.24 (exog variables ile)
GARCH:   persistence=0.95+, vol_corr=0.3+
```

## ONNX Export Sorunu

`skl2onnx` XGBoost/LightGBM/CatBoost için converter bulamıyor.
Çözüm: Native ONNX export kullan.

```bash
pip install xgboost-onnx  # XGBoost native ONNX
# veya
pip install onnxmltools    # Alternatif converter
```

`onnxscript` eksikse DL ONNX export başarısız olur:
```bash
pip install onnxscript
```

## Önemli Notlar

- `macro_features.py` Colab'a yüklenmezse makro feature'lar atlanır (non-critical)
- `SQ` (Block Inc.) yfinance'da delisted — otomatik atlanır
- Walk-forward 420 fold çalışır (~12 dakika) — `--skip-wf` ile atlanabilir
- IC Sharpe < 0.5 → alpha zayıf, macro features ekle
- IC Sharpe > 0.5 → consistent alpha var
