"""
FinAI AI Models Service — Production Inference
================================================
Pre-trained model serving with Redis caching.
NO on-the-fly training — uses model_registry artifacts.

Supports: XGBoost, LightGBM, CatBoost (joblib), LSTM/TCN/TFT (ONNX), ARIMA (pickle)
"""
import time
import logging
import hashlib
import warnings
warnings.filterwarnings('ignore')

from services.model_serving import get_model_server
from services.redis_cache import (
    cache_get, cache_set, prediction_key,
    compute_data_hash, get_ttl_for_prediction
)

logger = logging.getLogger(__name__)

# Valid model types for input validation
VALID_MODEL_TYPES = frozenset({
    "ensemble", "xgboost", "lightgbm", "catboost",
    "lstm", "tcn", "tft", "arima", "sarima", "sarimax"
})

MAX_HORIZON = 60   # PredictionRequest ile tutarlı (1-60)
MIN_DATA_POINTS = 30


def predict_next_price(market_data, model_type="ensemble", horizon=10, symbol="UNKNOWN"):
    """
    Run prediction using pre-trained models from the registry.

    Args:
        market_data: List of OHLCV dicts
        model_type: One of VALID_MODEL_TYPES
        horizon: Future bars to predict (1-60)
        symbol: Stock/crypto symbol

    Returns:
        Dict with prediction results or error
    """
    total_start = time.time()

    # ── Input validation ──
    if not isinstance(market_data, list) or len(market_data) < MIN_DATA_POINTS:
        return {"error": f"Need at least {MIN_DATA_POINTS} data points, got {len(market_data) if isinstance(market_data, list) else 0}"}

    model_type = model_type.lower().strip()
    if model_type not in VALID_MODEL_TYPES:
        return {"error": f"Invalid model_type '{model_type}'. Valid: {sorted(VALID_MODEL_TYPES)}"}

    horizon = max(1, min(int(horizon), MAX_HORIZON))
    symbol = symbol.upper().strip()

    # ── Check Redis cache ──
    data_hash = compute_data_hash(market_data)
    redis_key = prediction_key(symbol, model_type, "auto", horizon, data_hash)

    cached = cache_get(redis_key)
    if cached is not None:
        logger.info(f"[CACHE HIT] {symbol}/{model_type}/h{horizon}")
        cached["_cache"] = "hit"
        cached["_response_time_ms"] = round((time.time() - total_start) * 1000, 1)
        return cached

    # ── Run inference via ModelServer ──
    try:
        server = get_model_server()
        result = server.predict(
            symbol=symbol,
            market_data=market_data,
            model_type=model_type,
            horizon=horizon,
        )
    except Exception as e:
        logger.error(f"[PREDICTION FAILED] {symbol}/{model_type}: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

    if "error" in result:
        return result

    result["_cache"] = "miss"
    result["_response_time_ms"] = round((time.time() - total_start) * 1000, 1)

    # ── Cache in Redis ──
    ttl = get_ttl_for_prediction("auto")
    cache_set(redis_key, result, ttl)
    logger.info(f"[CACHED] {symbol}/{model_type}/h{horizon} TTL={ttl}s t={result['_response_time_ms']}ms")

    return result
