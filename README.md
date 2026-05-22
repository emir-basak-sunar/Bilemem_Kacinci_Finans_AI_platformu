# FinAI Platform

Kurumsal düzeyde finansal yapay zeka platformu. Hisse senedi ve kripto para piyasaları için çok katmanlı ML tahmin motoru, gerçek zamanlı piyasa verisi, temel analiz ve portföy yönetimi araçlarını tek bir platformda birleştirir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Teknoloji Yığını](#teknoloji-yığını)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Servisler](#servisler)
- [AI / ML Pipeline](#ai--ml-pipeline)
- [Model Kataloğu](#model-kataloğu)
- [Feature Engineering](#feature-engineering)
- [API Referansı](#api-referansı)
- [Feature Store](#feature-store)
- [Piyasa Rejimi Tespiti](#piyasa-rejimi-tespiti)
- [Metrikler ve Değerlendirme](#metrikler-ve-değerlendirme)
- [Geliştirme Notları](#geliştirme-notları)

---

## Genel Bakış

FinAI Platform, birden fazla ML modelini (XGBoost, LightGBM, CatBoost, LSTM, TCN, TFT, ARIMA, SARIMA, SARIMAX) ensemble yöntemiyle birleştirerek fiyat tahmini üretir. Sistem hedge fon standartlarında tasarlanmıştır:

- **160+ teknik, makro ve cross-sectional feature** ile eğitilmiş modeller
- **Walk-forward validation** ile leakage-free backtest
- **Purged K-Fold** ile zaman serisi için doğru cross-validation
- **Piyasa rejimi tespiti** (Gaussian HMM) ile dinamik ensemble ağırlıkları
- **Feature Store** (Redis online + Parquet offline) ile düşük gecikmeli serving
- **MLflow** entegrasyonu ile model versiyonlama ve experiment tracking
- **JWT tabanlı kimlik doğrulama**, abonelik planları ve kullanım kotası

---

## Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│              localhost:3000  —  TypeScript + Tailwind            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (authFetch)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Spring Boot Gateway (Java 21)                   │
│   localhost:8080  —  Auth · Rate Limit · Quota · Cache · Proxy  │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  Auth    │  │  Market  │  │  AI      │  │  Feature Store │  │
│  │  /auth   │  │  /market │  │  /ai     │  │  /feature-store│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebClient (HTTP proxy)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Python FastAPI Backend                          │
│   localhost:8000  —  Model Serving · Feature Engineering        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ ModelServer  │  │ FeatureStore │  │  MarketData Service  │  │
│  │ XGB/LGB/CAT  │  │ Redis+Parq.  │  │  Yahoo Finance API   │  │
│  │ LSTM/TCN/TFT │  └──────────────┘  └──────────────────────┘  │
│  │ ARIMA family │                                                │
│  └──────────────┘                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐    ┌─────────────────────────┐
│   PostgreSQL 15      │    │   Redis 7               │
│   localhost:5433     │    │   localhost:6379        │
│   Users · Wallet     │    │   Cache · Sessions      │
│   Subscriptions      │    │   Feature Store Online  │
│   Audit Logs         │    │   Rate Limiting         │
└─────────────────────┘    └─────────────────────────┘
```

---

## Teknoloji Yığını

### Frontend
| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| Next.js | 15 | App Router, SSR |
| TypeScript | 5 | Tip güvenliği |
| Tailwind CSS | 3 | Stil |
| Zustand | 4 | Global state (auth, market, settings) |
| Lightweight Charts | 4 | TradingView tarzı finansal grafikler |
| Recharts | 2 | Dashboard grafikleri |

### Spring Boot Gateway
| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| Java | 21 | Runtime |
| Spring Boot | 3.2 | Framework |
| Spring Security | 6 | JWT auth, rate limiting |
| Spring WebFlux | 6 | Reactive HTTP proxy (WebClient) |
| PostgreSQL | 15 | Ana veritabanı |
| Redis | 7 | Cache, session, rate limit |
| Flyway | 9 | DB migration |
| Lombok | 1.18 | Boilerplate azaltma |

### Python Backend
| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| FastAPI | 0.111 | REST API |
| Python | 3.13 | Runtime |
| XGBoost | 2.x | Gradient boosting |
| LightGBM | 4.x | Gradient boosting |
| CatBoost | 1.x | Gradient boosting |
| PyTorch | 2.x | LSTM, TCN, TFT |
| ONNX Runtime | 1.x | Hızlı DL inference |
| pmdarima | 2.x | ARIMA/SARIMA/SARIMAX |
| scikit-learn | 1.x | Preprocessing, metrics |
| pandas / numpy | 2.x / 1.x | Veri işleme |
| ta | 0.11 | Teknik indikatörler |
| hmmlearn | 0.3 | Gaussian HMM (rejim tespiti) |
| MLflow | 2.x | Experiment tracking |
| Optuna | 3.x | Hyperparameter optimization |
| Redis-py | 5.x | Cache client |
| httpx | 0.27 | Async HTTP (Yahoo Finance) |

---

## Proje Yapısı

```
FinAI/
├── frontend/                    # Next.js uygulaması
│   └── src/
│       ├── app/                 # App Router sayfaları
│       │   ├── page.tsx         # Ana dashboard
│       │   └── settings/        # Kullanıcı ayarları
│       ├── components/
│       │   ├── AI/              # Model seçici, analiz modalları
│       │   ├── Auth/            # Login/register modal
│       │   ├── Chart/           # TradingView tarzı grafik
│       │   ├── FeatureStore/    # Feature store dashboard
│       │   ├── Fundamentals/    # Temel analiz paneli
│       │   └── ...
│       ├── lib/
│       │   └── api.ts           # Tüm API çağrıları (authFetch)
│       └── store/
│           ├── authStore.ts     # JWT token yönetimi
│           └── marketStore.ts   # Piyasa verisi state
│
├── spring-backend/              # Java Spring Boot gateway
│   └── src/main/java/com/finai/
│       ├── ai/                  # AI proxy, quota, PredictionRequest
│       ├── auth/                # JWT, refresh token
│       ├── market/              # Piyasa verisi proxy
│       ├── user/                # Kullanıcı profili
│       ├── wallet/              # Cüzdan, para yatırma/çekme
│       ├── subscription/        # Plan yönetimi
│       └── config/              # Security, CORS, WebClient
│
├── backend/                     # Python FastAPI backend
│   ├── main.py                  # FastAPI app, tüm endpoint'ler
│   └── services/
│       ├── ai_models.py         # Prediction orchestration + Redis cache
│       ├── model_serving.py     # ModelServer — model yükleme ve inference
│       ├── market_data.py       # Yahoo Finance async fetcher
│       ├── regime_detection.py  # Gaussian HMM rejim tespiti
│       ├── stock_fundamentals.py# yfinance temel analiz
│       ├── redis_cache.py       # Redis cache utilities
│       └── mlflow_service.py    # MLflow import/summary
│
├── training/                    # ML eğitim pipeline
│   ├── config.py                # Tüm hyperparameter ve path config
│   ├── data_collection.py       # yfinance veri toplama
│   ├── walk_forward.py          # PurgedKFold, WalkForward, quant metrics
│   ├── model_registry.py        # Model kayıt ve manifest yönetimi
│   ├── run_all.py               # Tüm modelleri sırayla eğit
│   ├── features/
│   │   ├── feature_engineering.py  # 160+ feature hesaplama
│   │   ├── feature_store.py        # Offline (Parquet) + Online (Redis) store
│   │   └── macro_features.py       # VIX, SP500, DXY, yield curve
│   └── models/
│       ├── tree/
│       │   └── train_tree_models.py # XGBoost/LightGBM/CatBoost eğitimi
│       ├── dl/
│       │   ├── architectures.py     # BiLSTM, TCN, TFT mimarileri
│       │   ├── train_lstm.py        # LSTM eğitimi
│       │   └── train_tcn_tft.py     # TCN ve TFT eğitimi
│       └── statistical/
│           ├── train_arima.py       # ARIMA/SARIMA eğitimi
│           └── train_garch.py       # GARCH volatilite modeli
│
├── trained_models/              # Eğitilmiş model artifacts
│   ├── manifests/               # Model manifest JSON'ları
│   ├── metrics/                 # Backtest ve SHAP metrikleri
│   └── models/                  # joblib, .pt, .onnx, .pkl dosyaları
│       ├── catboost/
│       ├── lightgbm/
│       ├── xgboost/
│       ├── lstm/
│       ├── tcn/
│       ├── tft/
│       ├── arima/
│       ├── sarima/
│       └── sarimax/
│
├── docker-compose.yml           # PostgreSQL + Redis
└── architecture.md              # Detaylı mimari dokümantasyon
```

---

## Kurulum

### Ön Gereksinimler

- Docker Desktop (PostgreSQL + Redis için)
- Java 21 (Spring Boot)
- Python 3.13
- Node.js 20+
- Maven 3.9+

### 1. Veritabanı ve Redis

```bash
docker-compose up -d
```

`docker-compose.yml` içeriği:
- PostgreSQL 15 → `localhost:5433` (kullanıcı: `finai`, şifre: `finai_secret`, db: `finai_db`)
- Redis 7 → `localhost:6379`

### 2. Spring Boot Gateway

```bash
cd spring-backend
mvn spring-boot:run
# veya
mvn clean package -DskipTests
java -jar target/finai-platform-*.jar
```

Uygulama `localhost:8080/api/v1` adresinde başlar.

Ortam değişkenleri (opsiyonel, varsayılanlar `application.yml`'de):
```bash
DB_USERNAME=finai
DB_PASSWORD=finai_secret
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET=<en az 64 karakter>
AI_SERVICE_URL=http://localhost:8000
```

### 3. Python Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Uygulama `localhost:8000` adresinde başlar. Swagger UI: `http://localhost:8000/docs`

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Uygulama `localhost:3000` adresinde başlar.

### 5. Eğitim (Opsiyonel — modeller zaten `trained_models/` içinde)

```bash
cd training
pip install -r requirements_training.txt

# Tüm modelleri eğit (Google Colab A100 önerilir)
python run_all.py

# Sadece tree modelleri
python models/tree/train_tree_models.py

# Sadece LSTM/TCN/TFT
python models/dl/train_lstm.py
python models/dl/train_tcn_tft.py

# Sadece ARIMA ailesi
python models/statistical/train_arima.py
```

---

## Servisler

### Spring Boot Gateway (`localhost:8080/api/v1`)

Tüm frontend istekleri bu gateway üzerinden geçer. Gateway şunları sağlar:

**Kimlik Doğrulama**
- `POST /auth/register` — Yeni kullanıcı kaydı
- `POST /auth/login` — JWT access + refresh token
- `POST /auth/refresh` — Token yenileme

**Piyasa Verisi**
- `GET /market/{symbol}?period=1mo` — OHLCV + teknik indikatörler

**AI Tahmin**
- `POST /ai/predict` — Model tahmini (quota kontrollü)
- `GET /ai/quota` — Kullanım kotası bilgisi

**Temel Analiz**
- `GET /fundamentals/{symbol}` — Tam temel analiz paketi
- `GET /fundamentals/{symbol}/overview` — Şirket özeti
- `GET /fundamentals/{symbol}/financials` — Finansal tablolar
- `GET /fundamentals/{symbol}/analyst` — Analist tavsiyeleri
- `GET /fundamentals/{symbol}/earnings` — Kazanç geçmişi
- `GET /fundamentals/{symbol}/news` — Haberler

**Kullanıcı**
- `GET /users/me` — Profil bilgisi
- `PUT /users/me` — Profil güncelleme
- `PUT /users/me/password` — Şifre değiştirme

**Cüzdan**
- `GET /wallet` — Bakiye
- `POST /wallet/deposit` — Para yatırma
- `POST /wallet/withdraw` — Para çekme

**Abonelik**
- `GET /plans` — Mevcut planlar
- `GET /subscriptions/me` — Aktif abonelik
- `POST /subscriptions` — Plan satın alma

**Feature Store**
- `POST /feature-store/materialize/{symbol}` — Feature hesapla ve kaydet
- `GET /feature-store/features/{symbol}` — Online feature'ları getir
- `GET /feature-store/features/{symbol}/stats` — Offline istatistikler
- `GET /feature-store/features/{symbol}/validate` — Feature drift kontrolü
- `GET /feature-store/registry` — Tüm feature view'ları
- `GET /feature-store/views` — Feature view listesi

**Rejim Tespiti**
- `GET /regime/{symbol}?period=3mo` — Gaussian HMM piyasa rejimi

**MLflow**
- `GET /mlflow/summary` — Tüm model metrikleri özeti
- `GET /mlflow/comparison` — Model karşılaştırması (IC + Sharpe)
- `GET /mlflow/run/{runName}` — Belirli model detayı
- `POST /mlflow/import` — Modelleri MLflow'a import et

### Python FastAPI Backend (`localhost:8000`)

Spring gateway'in proxy ettiği asıl AI servisi. Doğrudan da erişilebilir (geliştirme için).

**Sağlık**
- `GET /` — Servis durumu, yüklü model sayısı, Redis bağlantısı

**AI Tahmin**
- `POST /api/predict` — Model inference
- `GET /api/models` — Mevcut modeller

**Piyasa Verisi**
- `GET /api/market-data/{symbol}?period=1mo`

**Rejim**
- `GET /api/regime/{symbol}?period=3mo`

**Feature Store**
- `POST /api/features/materialize/{symbol}`
- `GET /api/features/{symbol}`
- `GET /api/features/{symbol}/stats`
- `GET /api/features/{symbol}/validate`

**MLflow**
- `POST /api/mlflow/import`
- `GET /api/mlflow/summary`
- `GET /api/mlflow/comparison`

---

## AI / ML Pipeline

### Veri Toplama

`training/data_collection.py` — Yahoo Finance üzerinden günlük OHLCV verisi çeker.

**Equity Universe (22 sembol):**
AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, IBM, META, NFLX, AMD, INTC, JPM, GS, BAC, V, MA, DIS, PYPL, SQ, COIN, UBER, CRM, ADBE, ORCL, CSCO, QCOM, AVGO

**Kripto (ayrı model):**
BTC-USD, ETH-USD

Kripto ve hisse senetleri kasıtlı olarak ayrı modellerde eğitilir — kripto 7/24 işlem görür, volatilite 5-10x daha yüksektir ve farklı dinamiklere sahiptir.

### Feature Engineering

`training/features/feature_engineering.py` — 160+ feature hesaplar.

**Feature Kategorileri:**

| Kategori | Feature Sayısı | Örnekler |
|----------|---------------|----------|
| Price Action | 13 | return_1d, volatility_20d, gk_volatility, gap |
| Trend | 26 | close_to_sma_200, macd_norm, adx, ichimoku |
| Momentum | 8 | rsi_14, stoch_k, williams_r, cci_20 |
| Volatility | 7 | bb_width, bb_position, atr_pct, parkinson_vol |
| Volume | 7 | obv_change, mfi_14, volume_ratio, close_to_vwap |
| Statistical | 5 | skewness_20, kurtosis_20, zscore_50, autocorr_5 |
| Microstructure | 12 | amihud_illiq, order_flow_imbalance, efficiency_ratio |
| Lags | 14 | return_lag_1..20, close_return_lag_1..10 |
| Calendar | 5 | day_of_week, month, quarter, is_month_end |
| Macro | ~15 | vix, sp500_return, yield_curve, dxy, gold |
| Cross-sectional | ~20 | rank features (hedge fund alpha framework) |

**Önemli Düzeltmeler (v2.0):**
- `target_vol` leakage fix: gelecek barları kullanmıyor
- Autocorrelation O(n²) → O(n) rolling corr
- Ham fiyat seviyesi feature'ları (sma_200, ema_200) model input'undan çıkarıldı — normalize versiyonları (close_to_sma_200) kullanılıyor
- Cross-symbol sequence sızıntısı önlendi

### Model Eğitimi

**Tree Modeller** (`training/models/tree/train_tree_models.py`):
1. Her sembol için ayrı time-series split (%80 train / %10 val / %10 test)
2. XGBoost, LightGBM, CatBoost paralel eğitimi
3. Optuna ile hyperparameter tuning (SQLite persistence)
4. Inverse-RMSE ağırlıklı ensemble
5. SHAP feature importance hesaplama
6. ONNX export

**Derin Öğrenme** (`training/models/dl/`):
- **BiLSTM**: 3 katman, 256 hidden, bidirectional, dropout 0.2
- **TCN**: 8 katman, receptive field ~1020 bar, kernel_size=3
- **TFT**: Temporal Fusion Transformer, 7 quantile çıktısı (P5-P95)

**İstatistiksel Modeller** (`training/models/statistical/`):
- **ARIMA**: pmdarima auto_arima, stepwise search
- **SARIMA**: Mevsimsel ARIMA (m=5, haftalık)
- **SARIMAX**: Makro değişkenlerle SARIMA
- **GARCH**: Volatilite modeli

### Walk-Forward Validation

`training/walk_forward.py` — Lopez de Prado "Advances in Financial Machine Learning" referans alınarak implement edilmiştir.

**PurgedKFold:**
- Test setine yakın train sample'ları purge eder (label overlap önleme)
- Embargo: test sonrası ek gap
- `horizon` parametresi dinamik — target_10 için purge=10 bar

**WalkForwardValidator:**
- Expanding window: her fold'da tüm geçmiş veri ile eğitim
- `test_window=63` bar (~1 çeyrek)
- IC Sharpe hesabı: `mean(IC) / std(IC) * sqrt(252/test_window)`

**Quant Metrics (v2.1 — düzeltilmiş):**
- Sharpe: `mean(ret) / std(ret) * sqrt(252/horizon)` — horizon-adjusted
- IC: stride=horizon ile overlap-adjusted Spearman rank correlation
- Long-short: top/bottom tercile portfolio, piyasa nötr alpha

### Model Registry

`training/model_registry.py` — Her eğitilmiş model için manifest JSON kaydeder.

Manifest içeriği:
```json
{
  "symbol": "EQUITY_H10",
  "version": 1,
  "data_range": ["2019-01-01", "2024-12-31"],
  "feature_columns": ["return_1d", "rsi_14", ...],
  "ensemble_weights": {"xgboost": 0.34, "lightgbm": 0.33, "catboost": 0.33},
  "backtest_metrics": {"ic": 0.49, "sharpe": 1.15, "directional_accuracy": 0.67},
  "horizons": [1, 5, 10, 20]
}
```

---

## Model Kataloğu

Sistemde 22 adet eğitilmiş model ailesi bulunur. Her aile XGBoost + LightGBM + CatBoost içerir.

### Equity Modelleri

| Model | Horizon | Hedef | Directional Acc | Sharpe (düz.) |
|-------|---------|-------|-----------------|---------------|
| EQUITY_H5_v1/v2 | 5 gün | 5 günlük return | ~67% | ~1.6 |
| EQUITY_H10_v1/v2 | 10 gün | 10 günlük return | ~67% | ~1.15 |
| EQUITY_H20_v1/v2 | 20 gün | 20 günlük return | ~67% | ~0.48 |
| EQUITY_CS_RANK_v1/v2 | 1 gün | Cross-sectional rank | ~58% | ~1.8 |
| EQUITY_UNIVERSAL_v1/v2 | 1 gün | 1 günlük return | ~55% | ~1.5 |

### Sektör Modelleri

| Model | Semboller |
|-------|-----------|
| SECTOR_TECH_v1/v2 | AAPL, MSFT, GOOGL, NVDA, AMD, ... |
| SECTOR_FINANCE_v1/v2 | JPM, GS, BAC, V, MA |
| SECTOR_CONSUMER_v1/v2 | AMZN, TSLA, NFLX, DIS, UBER |
| SECTOR_OTHER_v1/v2 | IBM, PYPL, SQ, COIN |

### Kripto Modelleri

| Model | Semboller | Directional Acc | Sharpe (düz.) |
|-------|-----------|-----------------|---------------|
| CRYPTO_UNIVERSAL_v1/v2 | BTC-USD, ETH-USD | ~55% | ~1.5 |

### Manifest Çözümleme Zinciri

Bir sembol için tahmin istendiğinde model şu sırayla seçilir:

```
1. EQUITY_H{5/10/20}  (horizon'a göre)
2. SECTOR_{TECH/FINANCE/CONSUMER/OTHER}
3. EQUITY_UNIVERSAL
4. CRYPTO_UNIVERSAL  (kripto için)
5. UNIVERSAL         (fallback)
```

---

## Feature Engineering

### Inference'ta Feature Hesaplama

Model serving sırasında `build_features()` çağrılır:

```python
# model_serving.py
df_featured = build_features(df_ohlcv, drop_na=False)
df_featured = df_featured.fillna(0)  # NaN → 0 (inference için güvenli)
```

**Önemli Not:** `drop_na=False` inference için kritik. `drop_na=True` eğitimde kullanılır (warmup satırlarını temizler). Inference'ta son satır kullanıldığı için warmup drop'u gerekmez.

### Cross-Sectional Features

Hedge fon alpha framework'ü: mutlak return yerine semboller arası rank kullanılır.

```python
# Her zaman adımında, her sembolün feature değeri diğer sembollerle karşılaştırılır
# rank(feature) / n_symbols → [0, 1] arası normalize rank
```

Bu yaklaşım piyasa nötr alpha üretir — piyasa yönünden bağımsız.

---

## API Referansı

### POST /api/v1/ai/predict

```json
{
  "symbol": "AAPL",
  "model_type": "ensemble",
  "period": "1mo",
  "horizon": 10
}
```

**model_type değerleri:** `ensemble`, `xgboost`, `lightgbm`, `catboost`, `lstm`, `tcn`, `tft`, `arima`, `sarima`, `sarimax`

**period değerleri:** `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`

**horizon:** 1-60 (gün)

**Yanıt:**
```json
{
  "current_price": 185.50,
  "prediction": -0.0123,
  "models": {
    "xgboost": -0.0145,
    "lightgbm": -0.0098,
    "catboost": -0.0134,
    "lstm": -0.0089,
    "arima": 0.0021
  },
  "future_path": [
    {"time": 1748000000, "value": 183.22},
    {"time": 1748086400, "value": 181.05}
  ],
  "confidence_intervals": {
    "p10": -0.045,
    "p25": -0.025,
    "p50": -0.012,
    "p75": 0.008,
    "p90": 0.021
  },
  "model_version": 1,
  "manifest_used": "EQUITY_H10",
  "training_metrics": {
    "ic": 0.49,
    "directional_accuracy": 0.67,
    "sharpe": 1.15
  }
}
```

**Not:** `prediction` değeri **return** (yüzde değişim) olarak döner, fiyat değil. Örneğin `-0.0123` = `%1.23 düşüş beklentisi`.

### GET /api/v1/regime/{symbol}

```json
{
  "symbol": "AAPL",
  "regime": "bull_low_vol",
  "confidence": 0.87,
  "vix": 14.2,
  "description": "Düşük volatilite, yükseliş trendi",
  "ensemble_weights": {
    "xgboost": 0.30,
    "lightgbm": 0.30,
    "catboost": 0.20,
    "lstm": 0.20
  }
}
```

**Rejim değerleri:** `bull_low_vol`, `bull_high_vol`, `bear_low_vol`, `bear_high_vol`, `crisis`

---

## Feature Store

FEAST mimarisinden ilham alınan hafif feature store implementasyonu.

### Bileşenler

**FeatureRegistry** — Feature view tanımlarını JSON'da saklar (`training/feature_store_data/registry/`)

**OfflineStore** — Parquet tabanlı tarihsel feature storage (`training/feature_store_data/offline/{SYMBOL}/{period}.parquet`)

**OnlineStore** — Redis tabanlı düşük gecikmeli serving (`db=1`, key: `fs:online:{SYMBOL}:{view_name}`)

### Feature View'lar

| View | Feature Sayısı | TTL | Açıklama |
|------|---------------|-----|----------|
| price_action | 13 | 5 dk | Return, volatilite, fiyat pozisyonu |
| trend | 26 | 1 saat | SMA/EMA, MACD, ADX, Ichimoku |
| momentum | 8 | 5 dk | RSI, Stochastic, Williams %R |
| volatility | 7 | 5 dk | Bollinger, ATR, Parkinson |
| volume | 7 | 5 dk | OBV, MFI, VWAP |
| statistical | 5 | 1 saat | Skewness, kurtosis, z-score |
| lags | 14 | 5 dk | Gecikmeli return ve hacim |
| calendar | 5 | 1 gün | Takvim özellikleri |
| symbol_meta | 1 | 1 gün | Symbol encoding |

### Kullanım

```bash
# Materialize (feature hesapla ve kaydet)
POST /api/v1/feature-store/materialize/AAPL?period=1y

# Online features getir (inference için)
GET /api/v1/feature-store/features/AAPL

# Offline istatistikler
GET /api/v1/feature-store/features/AAPL/stats

# Model ile feature uyumunu kontrol et
GET /api/v1/feature-store/features/AAPL/validate
```

---

## Piyasa Rejimi Tespiti

`backend/services/regime_detection.py` — Gaussian Hidden Markov Model (HMM) kullanır.

### Rejimler

| Rejim | VIX | SP500 Trendi | Açıklama |
|-------|-----|-------------|----------|
| bull_low_vol | < 15 | > MA200 | Normal yükseliş |
| bull_high_vol | 15-25 | > MA200 | Volatil yükseliş |
| bear_low_vol | < 25 | < MA200 | Düşüş başlangıcı |
| bear_high_vol | 25-40 | < MA200 | Aktif düşüş |
| crisis | > 40 | herhangi | Kriz modu |

### Dinamik Ensemble Ağırlıkları

Her rejim için farklı model ağırlıkları kullanılır:

```python
"bull_low_vol":  {"xgboost": 0.30, "lightgbm": 0.30, "catboost": 0.20, "lstm": 0.20}
"bear_high_vol": {"lstm": 0.40, "tft": 0.30, "xgboost": 0.20, "lightgbm": 0.10}
"crisis":        {"lstm": 0.50, "tft": 0.30, "xgboost": 0.20}
```

---

## Metrikler ve Değerlendirme

### IC (Information Coefficient)

Tahmin ile gerçek return arasındaki Spearman rank korelasyonu.

| IC Değeri | Yorum |
|-----------|-------|
| 0.00 – 0.02 | Zayıf |
| 0.02 – 0.05 | Kullanılabilir |
| 0.05 – 0.10 | İyi |
| 0.10 – 0.15 | Çok iyi |
| 0.15+ | Şüpheli (leakage kontrolü gerekir) |

**Not:** Bu sistemdeki IC değerleri (~0.49) overlapping target'lardan kaynaklanır. `target_10` için 10 bar overlap vardır — bağımsız sample sayısı gerçekte ~1/10'dur. Gerçek IC ~0.05-0.12 aralığında olduğu tahmin edilmektedir.

### Sharpe Oranı (Düzeltilmiş)

`Sharpe = mean(ret) / std(ret) * sqrt(252 / horizon)`

Horizon-adjusted annualization kullanılır. `horizon=10` için `sqrt(252/10) ≈ 5.02` (eski `sqrt(252) ≈ 15.87` değil).

### Directional Accuracy

Yön tahmini doğruluğu — en güvenilir metrik (overlap'tan etkilenmez).

- %50: rastgele tahmin
- %54+: kullanılabilir
- %60+: güçlü sinyal
- Bu sistemde: **~%67** (güvenilir)

---

## Geliştirme Notları

### Bilinen Sınırlamalar

1. **IC Şişmesi:** Overlapping target'lar (target_10 için 10 bar overlap) IC değerini şişirir. Gerçek IC için yeniden eğitim ve stride=horizon ile hesaplama gerekir.

2. **Survivorship Bias:** Eğitim universe'i mevcut büyük şirketlerden oluşur — tarihsel olarak başarısız olan şirketler dahil değil.

3. **Transaction Cost:** Backtest'te %0.1 round-trip maliyet varsayılmıştır. Gerçek maliyetler daha yüksek olabilir.

4. **Makro Feature'lar:** VIX, SP500, DXY verileri yfinance üzerinden çekilir. Bağlantı sorunu olduğunda makro feature'lar 0 ile doldurulur.

### Önemli Konfigürasyon

`training/config.py` — Tüm hyperparameter ve path ayarları buradadır.

```python
# Minimum bar sayısı (sma_200 warmup için)
# Inference'ta drop_na=False kullan, fillna(0) ile doldur
# Training'de drop_na=True kullan

# Period önerileri:
# - Predict: minimum "6mo" (otomatik yükseltilir)
# - Materialize: minimum "1y" (otomatik yükseltilir)
# - Regime: minimum "3mo"
```

### Backend Restart

Python backend değişikliklerinin aktif olması için restart gerekir:

```bash
# Mevcut process'i durdur (Ctrl+C)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` flag'i ile dosya değişikliklerinde otomatik restart olur (geliştirme için).

### Redis Cache TTL'leri

| Veri Tipi | TTL |
|-----------|-----|
| Piyasa verisi (1d/5d) | 5 dakika |
| Piyasa verisi (1mo) | 15 dakika |
| Piyasa verisi (3mo+) | 1 saat |
| AI tahmin | 30 dakika |
| Rejim tespiti | 30 dakika |
| Feature store (price_action) | 5 dakika |
| Feature store (trend/statistical) | 1 saat |
| Feature store (calendar/meta) | 1 gün |

### Abonelik Planları ve Kotalar

| Plan | Aylık Tahmin Kotası |
|------|---------------------|
| FREE | 10 |
| BASIC | Plan config'e göre |
| PRO | Plan config'e göre |
| ENTERPRISE | Plan config'e göre |

Cache hit'leri kota tüketmez — aynı parametrelerle yapılan tahminler 30 dakika boyunca ücretsizdir.

---

## Lisans

Bu proje özel kullanım içindir.
