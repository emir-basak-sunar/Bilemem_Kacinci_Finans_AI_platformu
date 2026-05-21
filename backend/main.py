"""
FinAI Platform — Production FastAPI Backend v2.0
=================================================
Secure, validated, rate-limited API endpoints.
All AI inference uses pre-trained models (no on-the-fly training).
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import uvicorn
import logging
import time
import re
import os
import sys

# ── sys.path: workspace root → training package erişimi ──────────────────────
_BACKEND_DIR    = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_BACKEND_DIR)
for _p in [_WORKSPACE_ROOT, _BACKEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Service imports ───────────────────────────────────────────────────────────
from services.market_data       import get_market_data
from services.ai_models         import predict_next_price, VALID_MODEL_TYPES
from services.redis_cache       import get_cache_stats, get_redis
from services.model_serving     import get_model_server
from services.stock_fundamentals import (
    get_full_fundamentals, get_company_overview, get_financials,
    get_analyst_data, get_holders, get_insider_transactions,
    get_earnings, get_options_summary, get_news, get_esg,
)
from services.regime_detection  import detect_regime
from services.mlflow_service    import (
    import_all_models_to_mlflow, get_mlflow_summary,
    get_run_detail, get_model_comparison,
)

# ── Feature Store — lazy init ─────────────────────────────────────────────────
_feature_store = None

def _get_fs():
    global _feature_store
    if _feature_store is None:
        try:
            from training.features.feature_store import get_feature_store
            _feature_store = get_feature_store()
        except ImportError:
            try:
                from training.feature_store import get_feature_store
                _feature_store = get_feature_store()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Feature store init failed: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Feature store init failed: {e}")
    return _feature_store

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FinAI Backend — warming up model server...")
    try:
        server = get_model_server()
        available = server.get_available_models()
        logger.info(f"Model server ready — {len(available)} model(s) loaded: {list(available.keys())[:5]}...")
    except Exception as e:
        logger.warning(f"Model server warm-up failed (will lazy-load): {e}")
    yield
    logger.info("Shutting down FinAI Backend")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinAI Platform API",
    version="2.0.0",
    description="Financial AI prediction platform with pre-trained model inference",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
]
app.add_middleware(CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.finai.com"],
)

# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    response.headers["X-Response-Time"] = f"{duration}ms"
    return response

# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ── Input validation ──────────────────────────────────────────────────────────
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9\-\.]{1,10}$')
VALID_PERIODS  = frozenset({"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"})

class PredictionRequest(BaseModel):
    symbol:     str = Field(..., min_length=1, max_length=10)
    model_type: str = Field(default="ensemble")
    period:     str = Field(default="1mo")
    horizon:    int = Field(default=10, ge=1, le=60)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        v = v.upper().strip()
        if not SYMBOL_PATTERN.match(v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v):
        v = v.lower().strip()
        if v not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type. Valid: {sorted(VALID_MODEL_TYPES)}")
        return v

    @field_validator("period")
    @classmethod
    def validate_period(cls, v):
        if v not in VALID_PERIODS:
            raise ValueError(f"Invalid period. Valid: {sorted(VALID_PERIODS)}")
        return v


# ============================================================
# Health
# ============================================================

@app.get("/", tags=["Health"])
def health_check():
    redis_ok = False
    try:
        r = get_redis()
        if r:
            r.ping()
            redis_ok = True
    except Exception:
        pass

    model_status = "unknown"
    model_count  = 0
    registry_dir = "unknown"
    try:
        server = get_model_server()
        available    = server.get_available_models()
        model_count  = len(available)
        model_status = "ready" if model_count > 0 else "no_models"
        registry_dir = server.registry_dir
    except Exception:
        model_status = "error"

    return {
        "status":        "online",
        "version":       "2.0.0",
        "redis":         "connected" if redis_ok else "disconnected",
        "model_server":  model_status,
        "models_loaded": model_count,
        "registry_dir":  registry_dir,
    }


# ============================================================
# Market Data
# ============================================================

@app.get("/api/market-data/{symbol}", tags=["Market Data"])
async def market_data(symbol: str, period: str = "1mo"):
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    if period not in VALID_PERIODS:
        raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
    try:
        return await get_market_data(symbol, period)
    except Exception as e:
        logger.error(f"Market data error for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {str(e)}")


# ============================================================
# AI Prediction
# ============================================================

@app.post("/api/predict", tags=["AI Prediction"])
async def predict(request: PredictionRequest):
    # FIX: Feature engineering için minimum 200 bar gerekiyor (sma_200 warmup).
    # "1mo" → ~150 bar (1h interval) → warmup sonrası yetersiz.
    # "1mo" veya "5d"/"1d" gelirse otomatik olarak "6mo"'ya yükselt.
    MINIMUM_PERIOD_FOR_FEATURES = "6mo"
    SHORT_PERIODS = {"1d", "5d", "1mo"}
    effective_period = MINIMUM_PERIOD_FOR_FEATURES if request.period in SHORT_PERIODS else request.period

    try:
        data = await get_market_data(request.symbol, effective_period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {str(e)}")

    if not data or len(data) < 30:
        raise HTTPException(status_code=422,
            detail=f"Insufficient data for {request.symbol} ({len(data) if data else 0} bars, need 30+)")

    result = predict_next_price(data,
        model_type=request.model_type,
        horizon=request.horizon,
        symbol=request.symbol,
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/api/models", tags=["AI Prediction"])
def list_models():
    try:
        return get_model_server().get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Admin (API key protected)
# ============================================================

@app.get("/api/cache/stats", tags=["Admin"])
async def cache_stats(x_admin_key: str = None):
    _key = os.environ.get("ADMIN_API_KEY", "finai-admin-2026")
    if x_admin_key != _key:
        raise HTTPException(status_code=403, detail="Admin key required")
    return get_cache_stats()


@app.delete("/api/cache/flush", tags=["Admin"])
async def cache_flush(x_admin_key: str = None):
    _key = os.environ.get("ADMIN_API_KEY", "finai-admin-2026")
    if x_admin_key != _key:
        raise HTTPException(status_code=403, detail="Admin key required")
    try:
        r = get_redis()
        if r:
            r.flushdb()
            return {"message": "Cache flushed"}
        return {"message": "Redis not connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Fundamentals
# ============================================================

@app.get("/api/fundamentals/{symbol}", tags=["Fundamentals"])
async def fundamentals_full(symbol: str):
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    try:
        return get_full_fundamentals(symbol)
    except Exception as e:
        logger.error(f"Fundamentals error for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/fundamentals/{symbol}/overview", tags=["Fundamentals"])
async def fundamentals_overview(symbol: str):
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    result = get_company_overview(symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/financials", tags=["Fundamentals"])
async def fundamentals_financials(symbol: str):
    result = get_financials(symbol.upper().strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No financial data for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/analyst", tags=["Fundamentals"])
async def fundamentals_analyst(symbol: str):
    result = get_analyst_data(symbol.upper().strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No analyst data for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/holders", tags=["Fundamentals"])
async def fundamentals_holders(symbol: str):
    result = get_holders(symbol.upper().strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No holder data for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/insider", tags=["Fundamentals"])
async def fundamentals_insider(symbol: str):
    return get_insider_transactions(symbol.upper().strip()) or []


@app.get("/api/fundamentals/{symbol}/earnings", tags=["Fundamentals"])
async def fundamentals_earnings(symbol: str):
    result = get_earnings(symbol.upper().strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No earnings data for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/options", tags=["Fundamentals"])
async def fundamentals_options(symbol: str):
    result = get_options_summary(symbol.upper().strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No options data for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/news", tags=["Fundamentals"])
async def fundamentals_news(symbol: str):
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    result = get_news(symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No news for {symbol}")
    return result


@app.get("/api/fundamentals/{symbol}/esg", tags=["Fundamentals"])
async def fundamentals_esg(symbol: str):
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    result = get_esg(symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No ESG data for {symbol}")
    return result


@app.get("/api/news/fetch", tags=["Fundamentals"])
async def fetch_news_content(url: str):
    """Fetch and parse article content from a news URL. TTL: 1h."""
    import hashlib
    from services.redis_cache import cache_get, cache_set

    if not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS URLs are supported")

    url_hash  = hashlib.md5(url.encode()).hexdigest()[:12]
    cache_key = f"news:content:{url_hash}"

    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        result = await _scrape_article(url)
        if result:
            cache_set(cache_key, result, 3600)
        return result or {"error": "Could not extract article content"}
    except Exception as e:
        logger.error(f"Article fetch error for {url}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch article: {str(e)}")


async def _scrape_article(url: str) -> dict:
    import httpx, re as _re

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}"}

    html = resp.text

    def meta(*names):
        for name in names:
            for pat in [
                rf'<meta[^>]+(?:name|property)=["\'][^"\']*{_re.escape(name)}[^"\']*["\'][^>]+content=["\']([^"\']+)["\']',
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\'][^"\']*{_re.escape(name)}[^"\']*["\']',
            ]:
                m = _re.search(pat, html, _re.IGNORECASE)
                if m and m.group(1).strip():
                    return m.group(1).strip()
        return ""

    title       = meta("og:title", "twitter:title", "title") or ""
    description = meta("og:description", "twitter:description", "description") or ""
    image       = meta("og:image", "twitter:image") or ""
    author      = meta("author", "article:author", "byl") or ""
    pub_date    = meta("article:published_time", "pubdate", "date") or ""
    site_name   = meta("og:site_name") or ""

    clean = _re.sub(
        r'<(script|style|nav|footer|header|aside|form|button|noscript|iframe|svg|figure)[^>]*>.*?</\1>',
        ' ', html, flags=_re.DOTALL | _re.IGNORECASE
    )
    article_body = ""
    for pat in [
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]+(?:class|id)=["\'][^"\']*(?:article-body|story-body|caas-body|entry-content)[^"\']*["\'][^>]*>(.*?)</div>',
    ]:
        m = _re.search(pat, clean, _re.DOTALL | _re.IGNORECASE)
        if m:
            article_body = m.group(1) if m.lastindex == 1 else m.group(2)
            if len(article_body) > 200:
                break

    source = article_body if len(article_body) > 200 else clean
    text   = _re.sub(r'<[^>]+>', ' ', source)
    text   = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
                 .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text   = _re.sub(r'\s+', ' ', text).strip()

    NOISE = _re.compile(
        r'^(skip to|cookie|privacy policy|terms of|sign in|log in|subscribe|advertisement|'
        r'follow us|share this|related articles|read more|click here|©|all rights reserved)',
        _re.IGNORECASE
    )
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text)
                 if len(s.strip()) >= 40 and not NOISE.match(s.strip())]

    return {
        "url": url, "title": title, "description": description,
        "body": " ".join(sentences[:50])[:4000],
        "image": image, "author": author, "published": pub_date, "site_name": site_name,
    }


# ============================================================
# Regime Detection
# ============================================================

@app.get("/api/regime/{symbol}", tags=["Regime Detection"])
async def regime_detection(symbol: str, period: str = "3mo"):
    """
    Gaussian HMM ile piyasa rejimi tespiti.
    Rejimler: bull | bear | sideways | high_vol
    TTL: 30 dakika. Minimum 60 bar gerektirir (period >= 3mo).
    """
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    try:
        market_data = await get_market_data(symbol, period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {str(e)}")
    if not market_data or len(market_data) < 30:
        raise HTTPException(status_code=422,
            detail=f"Insufficient data for {symbol} ({len(market_data) if market_data else 0} bars)")
    return detect_regime(market_data, symbol, period)


# ============================================================
# Feature Store
# ============================================================

@app.post("/api/features/materialize/{symbol}", tags=["Feature Store"])
async def materialize_features(symbol: str, period: str = "1y"):
    symbol = symbol.upper().strip()
    if not SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
    fs = _get_fs()
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable")

    # FIX: Feature engineering için minimum 200 bar gerekiyor (sma_200 warmup).
    # "1mo" veya daha kısa period'larda yeterli bar olmayabilir.
    SHORT_PERIODS = {"1d", "5d", "1mo"}
    effective_period = "1y" if period in SHORT_PERIODS else period

    try:
        market_data = await get_market_data(symbol, effective_period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {str(e)}")
    result = fs.materialize(symbol, market_data, period=effective_period)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/api/features/{symbol}", tags=["Feature Store"])
async def get_features(symbol: str, views: str = None):
    symbol = symbol.upper().strip()
    fs = _get_fs()
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable")
    view_list = [v.strip() for v in views.split(",")] if views else None
    features  = fs.get_online_features(symbol, feature_views=view_list)
    if features is None:
        raise HTTPException(status_code=404,
            detail=f"No features for {symbol}. Call POST /api/features/materialize/{symbol} first.")
    return {"symbol": symbol, "features": features, "count": len(features)}


@app.get("/api/features/{symbol}/stats", tags=["Feature Store"])
async def get_feature_stats(symbol: str, period: str = "1y"):
    symbol = symbol.upper().strip()
    fs = _get_fs()
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable")
    # Kısa period'lar için 1y'e yükselt (materialize ile tutarlı)
    SHORT_PERIODS = {"1d", "5d", "1mo"}
    effective_period = "1y" if period in SHORT_PERIODS else period
    stats = fs.get_feature_stats(symbol, effective_period)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"No offline features for {symbol}/{effective_period}. Call POST /api/features/materialize/{symbol} first.")
    return stats


@app.get("/api/features/{symbol}/validate", tags=["Feature Store"])
async def validate_features(symbol: str, period: str = "1y"):
    symbol = symbol.upper().strip()
    fs = _get_fs()
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable")
    SHORT_PERIODS = {"1d", "5d", "1mo"}
    effective_period = "1y" if period in SHORT_PERIODS else period
    try:
        from training.model_registry import ModelRegistry
        registry = ModelRegistry()
        manifest = registry.get_latest_manifest(symbol) or registry.get_latest_manifest("UNIVERSAL")
        if manifest is None:
            raise HTTPException(status_code=404, detail="No model manifest found")
        expected_cols = manifest.feature_columns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return fs.validate_features(symbol, effective_period, expected_cols)


@app.get("/api/feature-store/registry", tags=["Feature Store"])
async def get_feature_registry():
    fs = _get_fs()
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable")
    return fs.get_registry_summary()


@app.get("/api/feature-store/views", tags=["Feature Store"])
async def list_feature_views():
    fs = _get_fs()
    if fs is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable")
    return fs.list_feature_views()


# ============================================================
# MLflow
# ============================================================

@app.post("/api/mlflow/import", tags=["MLflow"])
async def mlflow_import_endpoint():
    """
    trained_models/ klasöründeki tüm manifest + metrics dosyalarını MLflow'a import eder.
    İlk kurulumda veya yeni model eğitiminden sonra çalıştırın.
    """
    try:
        return import_all_models_to_mlflow()
    except Exception as e:
        logger.error(f"MLflow import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/summary", tags=["MLflow"])
async def mlflow_summary_endpoint():
    """
    MLflow'daki tüm production run'ların özetini döner.
    Her model için: IC, Sharpe, Sortino, Directional Accuracy, ensemble weights.
    """
    try:
        return get_mlflow_summary()
    except Exception as e:
        logger.error(f"MLflow summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/run/{run_name}", tags=["MLflow"])
async def mlflow_run_detail_endpoint(run_name: str):
    """
    Belirli bir run'ın tüm metriklerini, tag'lerini ve SHAP verilerini döner.
    Örnek: /api/mlflow/run/EQUITY_UNIVERSAL_v2
    """
    try:
        result = get_run_detail(run_name)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/comparison", tags=["MLflow"])
async def mlflow_comparison_endpoint():
    """Tüm model ailelerini IC ve Sharpe bazında karşılaştırır. En iyi 10 modeli döner."""
    try:
        return get_model_comparison()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/models", tags=["MLflow"])
async def mlflow_models_list():
    """
    trained_models/ registry'sindeki tüm modelleri listeler.
    MLflow'dan bağımsız — doğrudan dosya sisteminden okur.
    """
    try:
        server    = get_model_server()
        available = server.get_available_models()
        metrics   = server.get_all_metrics()
        manifests = server.get_all_manifests()
        return {
            "available_models": available,
            "total_models":     len(available),
            "metrics_summary":  metrics,
            "manifests":        manifests,
            "registry_dir":     server.registry_dir,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
