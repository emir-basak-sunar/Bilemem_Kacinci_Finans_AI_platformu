# FinAI Training Pipeline — Analiz ve Değişiklik Raporu

**Son Güncelleme:** v2.0  
**Hedef:** Hedge fon kalitesi alpha — IC Sharpe > 0.5, Long-Short Sharpe > 1.0

---

## 1. v2.0 DEĞİŞİKLİKLERİ (Hedge Fund Grade)

### 1.1 Kritik Bug Fix'ler

| # | Sorun | Dosya | Durum |
|---|-------|-------|-------|
| 1 | `target_vol` leakage — `rolling(h)` gelecek barları kullanıyordu | `feature_engineering.py` | ✅ Düzeltildi |
| 2 | `autocorr` O(n²) — `rolling.apply(autocorr)` çok yavaş | `feature_engineering.py` | ✅ `rolling.corr` ile O(n) |
| 3 | Cross-symbol sequence sızıntısı — AAPL son barı + MSFT ilk barı aynı sequence | `train_dl_models.py` | ✅ `create_sequences_by_symbol` |
| 4 | SARIMA m=252 — saatler sürer, IC≈0 üretir | `train_statistical_models.py` | ✅ m=5 (haftalık) |
| 5 | Kripto + equity aynı model — pattern dilution | `train_tree_models.py` | ✅ Ayrı modeller |
| 6 | Optuna study persistence yok — session kapanınca kaybolur | `train_tree_models.py` | ✅ SQLite storage |
| 7 | Optuna arama alanı dar — min_child_weight, gamma eksik | `train_tree_models.py` | ✅ Genişletildi |
| 8 | PurgedKFold horizon sabit=1 — target_5 için yanlış purge | `walk_forward.py` | ✅ Dinamik horizon |

### 1.2 Yeni Özellikler

#### `feature_engineering.py` — Yeni Feature'lar
```
order_flow_imbalance  — (close-open)/(high-low): intraday alıcı/satıcı baskısı
order_flow_ma5        — 5-bar order flow ortalaması
realized_skew_20/60   — Realized skewness (crash risk indicator)
vol_regime_60_20      — 60-bar/20-bar vol oranı (rejim değişimi)
efficiency_ratio_20   — Hurst exponent proxy (trend vs random walk)
```

#### `feature_engineering.py` — Cross-Sectional Framework
```python
add_cross_sectional_features()  — rank-based features (hedge fund alpha)
  cs_rank_{feat}    — cross-sectional rank [-0.5, +0.5]
  cs_zscore_{feat}  — cross-sectional z-score

add_cross_sectional_targets()   — rank-based targets
  target_cs_rank_{h} — future return cross-sectional rank
  → IC'yi doğrudan optimize eder
```

#### `walk_forward.py` — Yeni Metrikler
```
compute_ic_sharpe()              — IC Sharpe = mean(IC)/std(IC)*sqrt(252/window)
WalkForwardValidator.validate_cross_sectional()  — multi-symbol IC
compute_quant_metrics()          — long-short strategy eklendi
RegimeDetector                   — VIX + SP500 trend ile rejim tespiti
```

#### `train_tree_models.py` — Yeni Pipeline
```
engineer_features(add_cs_features=True)  — cross-sectional features
_run_cross_sectional_validation()        — IC Sharpe hesabı
train_tree_models(target_col="target_cs_rank_1")  — IC-optimized model
Kripto ayrı model: CRYPTO_UNIVERSAL
Equity modeller: EQUITY_UNIVERSAL, EQUITY_H5/H10/H20, EQUITY_CS_RANK
```

#### `train_dl_models.py` — Yeni Pipeline
```
prepare_dl_data()  — create_sequences_by_symbol (cross-symbol fix)
_compute_regime_ensemble()  — equal + IC-weighted + regime-conditional
A100 optimize: hidden=256, layers=3, batch=256, seq_length=120
```

#### `train_statistical_models.py` — Yeni Modeller
```
train_garch_for_symbol()  — GARCH(1,1) volatility model
rolling_arima_forecast()  — expanding window backtest
```

---

## 2. METRİK REHBERİ v2.0

### 2.1 IC (Information Coefficient)
```
IC = Spearman(predicted_return, actual_return)

IC < 0.00  → noise
IC = 0.02  → weak alpha
IC = 0.05  → usable alpha (hedge fon minimum)
IC = 0.10+ → good alpha
IC = 0.20+ → exceptional (Renaissance Technologies seviyesi)
```

### 2.2 IC Sharpe (v2.0 — yeni kritik metrik)
```
IC Sharpe = mean(IC per fold) / std(IC per fold) * sqrt(252 / test_window)

IC Sharpe > 0.5  → consistent alpha (kullanılabilir)
IC Sharpe > 1.0  → strong alpha (iyi hedge fon)
IC Sharpe > 2.0  → exceptional (top-tier hedge fon)
```

### 2.3 Long-Short Sharpe (v2.0 — piyasa nötr)
```
Long-Short: top tercile long, bottom tercile short
→ Piyasa yönünden bağımsız alpha

Sharpe(LS) > 0.5  → usable
Sharpe(LS) > 1.0  → good
Sharpe(LS) > 1.5  → excellent
```

### 2.4 Quant Metrics Tablosu

| Metrik | Hesap | Hedef |
|--------|-------|-------|
| IC | Spearman(pred, actual) | > 0.05 |
| IC Sharpe | mean(IC)/std(IC)*sqrt(4) | > 0.5 |
| Sharpe (L) | annual_ret/annual_vol | > 1.0 |
| Sharpe (LS) | long-short annual_ret/vol | > 1.0 |
| Sortino | annual_ret/downside_vol | > 1.5 |
| Calmar | annual_ret/max_drawdown | > 1.0 |
| Max DD | peak-to-trough | < -20% |
| Turnover | avg daily position change | < 0.5 |

---

## 3. MİMARİ DEĞİŞİKLİKLERİ

### 3.1 Kripto Ayrımı
```
Önceki: BTC/ETH + 28 hisse = 1 universal model
Sorun: Kripto vol 5-10x hisse, 7/24 işlem, farklı dinamikler
       → Pattern dilution, IC düşük

Yeni:
  EQUITY_UNIVERSE (28 hisse) → EQUITY_UNIVERSAL model
  CRYPTO_SYMBOLS (BTC, ETH)  → CRYPTO_UNIVERSAL model (ayrı)
```

### 3.2 Cross-Sectional Alpha Framework
```
Önceki: Mutlak return tahmini (target_1)
Sorun: Piyasa yönüne bağımlı, IC düşük

Yeni: Cross-sectional rank tahmini (target_cs_rank_1)
  Her gün: tüm hisseler için return tahmin et
  → Sembolleri tahmine göre sırala
  → Top 3 long, Bottom 3 short
  → IC = Spearman(rank_pred, rank_actual)
  → Piyasa nötr alpha
```

### 3.3 Regime-Conditional Ensemble
```
Önceki: Sabit ağırlıklı ensemble (inverse-RMSE)
Sorun: Bull market'ta tree iyi, bear market'ta LSTM iyi

Yeni: RegimeDetector (VIX + SP500 trend)
  bull_low_vol:  tree ağırlıklı (XGB 30%, LGBM 30%, CAT 20%, LSTM 20%)
  bear_high_vol: LSTM ağırlıklı (LSTM 40%, TFT 30%, XGB 20%, LGBM 10%)
  crisis:        LSTM dominant (LSTM 50%, TFT 30%, XGB 20%)
```

---

## 4. KALAN SORUNLAR

### 🟡 Orta Öncelik

| # | Sorun | Etki |
|---|-------|------|
| 1 | Gerçek beta hesabı — SP500 returns ile rolling 60-bar beta | IC artışı |
| 2 | Options flow data — put/call ratio, IV skew | Alternative data |
| 3 | SEC EDGAR insider transactions | Alternative data |
| 4 | Kelly criterion position sizing | Risk management |
| 5 | Benchmark-relative alpha (SP500 excess return) | Gerçekçi PnL |

### 🟢 Düşük Öncelik

| # | Sorun | Etki |
|---|-------|------|
| 6 | model_registry.py weights_only=False güvenlik | Production security |
| 7 | Feature store online serving latency (9 Redis round-trip) | Inference speed |
| 8 | Leakage audit scripti | Model güvenilirliği |

---

## 5. COLAB A100 ÇALIŞTIRMA SIRASI

```bash
# 1. Kurulum
pip install -r training/requirements_training.txt

# 2. Tree modeller (veri toplama + feature engineering + eğitim)
python -m training.train_tree_models
# ~2-3 saat (Optuna 50 trial dahil)

# 3. DL modeller (A100 GPU)
python -m training.train_dl_models
# ~1-2 saat (LSTM + TCN + TFT)

# 4. Statistical modeller (CPU)
python -m training.train_statistical_models
# ~30 dakika (ARIMA + SARIMA + GARCH per symbol)
```

---

## 6. DOSYA BAĞIMLILIKLARI v2.0

```
config.py (EQUITY_UNIVERSE, CRYPTO_SYMBOLS, OptunaConfig, RegimeConfig)
    ↑
data_collection.py
    ↑
feature_engineering.py (add_cross_sectional_features, create_sequences_by_symbol)
    ↑
macro_features.py
    ↑
walk_forward.py (RegimeDetector, compute_ic_sharpe, validate_cross_sectional)
    ↑
train_tree_models.py  ← equity + kripto ayrı, CS rank target, Optuna SQLite
train_dl_models.py    ← by-symbol sequences, regime ensemble
train_statistical_models.py  ← GARCH, rolling forecast, m=5
    ↓
model_registry.py
    ↓
model_registry/ (manifests/ + models/ + metrics/)
```

---

*Rapor v2.0 — Hedge Fund Grade Pipeline*
