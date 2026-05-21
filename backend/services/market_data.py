"""
Market Data Service with Redis Caching
=======================================
Production-grade market data pipeline:
- Async HTTP via httpx (non-blocking)
- Redis cache with stampede prevention
- Dynamic TTL based on market hours
- Robust error handling and data cleaning
- Proper symbol normalization (BTC → BTC-USD)
"""
import asyncio
import httpx
import pandas as pd
import numpy as np
import time
import logging
from fastapi import HTTPException
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from services.redis_cache import (
    cache_get, cache_set, market_data_key,
    get_ttl_for_market_data, acquire_lock, release_lock
)

logger = logging.getLogger(__name__)

# ============================================================
# Symbol normalization
# ============================================================

SYMBOL_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "DOT": "DOT-USD",
    "BNB": "BNB-USD",
}

def normalize_symbol(symbol: str) -> str:
    """Map common ticker shortcuts to Yahoo Finance symbols."""
    s = symbol.upper().strip()
    return SYMBOL_MAP.get(s, s)


# ============================================================
# Interval selection strategy
# ============================================================

def get_interval_for_period(period: str) -> str:
    """
    Choose optimal candle interval based on the history range.
    
    Rules:
    - Yahoo free API limits 5m data to ~60 days
    - 1h data available for ~730 days
    - 1d data unlimited history
    """
    interval_map = {
        "1d":  "5m",    # ~78 bars — intraday
        "5d":  "15m",   # ~130 bars — multi-day intraday
        "1mo": "1h",    # ~150 bars — hourly for 1 month
        "3mo": "1d",    # ~63 bars — daily
        "6mo": "1d",    # ~126 bars — daily
        "ytd": "1d",    # variable — daily
        "1y":  "1d",    # ~252 bars — daily
        "2y":  "1wk",   # ~104 bars — weekly
        "5y":  "1wk",   # ~260 bars — weekly
        "10y": "1mo",   # ~120 bars — monthly
        "max": "1mo",   # variable — monthly
    }
    return interval_map.get(period, "1d")


# ============================================================
# Main public API
# ============================================================

async def get_market_data(symbol: str, period: str = "1mo"):
    """
    Get market data with Redis cache layer.
    
    Flow:
    1. Normalize symbol (BTC → BTC-USD)
    2. Check Redis cache → if HIT, return immediately
    3. If MISS → acquire lock → fetch from Yahoo → calculate indicators → cache → return
    4. If lock held by another request → wait → retry cache
    """
    yahoo_symbol = normalize_symbol(symbol)
    cache_key = market_data_key(symbol, period)  # Use original symbol for cache key consistency
    
    # ── Step 1: Try cache ──
    cached = cache_get(cache_key)
    if cached is not None:
        logger.info(f"[CACHE HIT] {symbol}/{period} — serving {len(cached)} bars from Redis")
        return cached
    
    # ── Step 2: Cache miss — acquire lock to prevent stampede ──
    logger.info(f"[CACHE MISS] {symbol}/{period} — fetching from Yahoo Finance")
    lock_acquired = acquire_lock(cache_key, timeout=15)
    
    if not lock_acquired:
        logger.info(f"[CACHE WAIT] {symbol}/{period} — another request is fetching, waiting...")
        await asyncio.sleep(0.5)  # Non-blocking wait
        cached = cache_get(cache_key)
        if cached is not None:
            return cached
        # Still no cache — proceed anyway
    
    try:
        # ── Step 3: Fetch from Yahoo Finance (async) ──
        start_time = time.time()
        result_data = await _fetch_and_process(yahoo_symbol, period)
        fetch_duration = round(time.time() - start_time, 3)
        
        # ── Step 4: Cache the result ──
        ttl = get_ttl_for_market_data(period)
        cache_set(cache_key, result_data, ttl)
        
        logger.info(
            f"[CACHED] {symbol}/{period} — {len(result_data)} bars, "
            f"TTL={ttl}s, fetch_time={fetch_duration}s"
        )
        
        return result_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market Data Error for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Data Provider Error: {str(e)}")
    finally:
        if lock_acquired:
            release_lock(cache_key)


# ============================================================
# Yahoo Finance data fetcher (async)
# ============================================================

async def _fetch_and_process(symbol: str, period: str) -> list:
    """
    Fetch OHLCV data from Yahoo Finance and compute technical indicators.
    Uses httpx for non-blocking HTTP.
    """
    interval = get_interval_for_period(period)
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": interval,
        "range": period,
        "includePrePost": "false",
        "events": "",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params, headers=headers)
    
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on Yahoo Finance.")
    
    if response.status_code != 200:
        logger.error(f"Yahoo API returned {response.status_code}: {response.text[:200]}")
        raise HTTPException(
            status_code=502,
            detail=f"Yahoo Finance API error (HTTP {response.status_code})"
        )
    
    data = response.json()
    
    chart_result = data.get("chart", {}).get("result")
    if not chart_result:
        error_msg = data.get("chart", {}).get("error", {}).get("description", "No data found")
        raise HTTPException(status_code=404, detail=f"No data for {symbol}: {error_msg}")
    
    result = chart_result[0]
    timestamps = result.get("timestamp")
    if not timestamps:
        raise HTTPException(status_code=404, detail=f"No price data available for {symbol}/{period}")
    
    quote = result["indicators"]["quote"][0]
    
    # ── Build clean OHLCV DataFrame ──
    df = pd.DataFrame({
        "time": timestamps,
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })
    
    # ── Data cleaning ──
    # Drop rows where OHLC is all NaN (market holidays, gaps)
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    
    # Forward-fill minor gaps (e.g., single missing bar)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    
    # Drop any remaining NaN rows
    df = df.dropna(subset=["open", "high", "low", "close"])
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No valid price data after cleaning.")
    
    # Fill missing volume with 0
    df["volume"] = df["volume"].fillna(0).astype(int)
    
    # Ensure timestamps are integers (Unix epoch)
    df["time"] = df["time"].astype(int)
    
    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)
    
    # Remove duplicate timestamps
    df = df.drop_duplicates(subset=["time"], keep="last")
    
    # ── Calculate Technical Indicators ──
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(df)

    # SMA 20
    df["sma"] = SMAIndicator(close=close, window=20).sma_indicator() if n >= 20 else None

    # EMA 20
    df["ema"] = EMAIndicator(close=close, window=20).ema_indicator() if n >= 20 else None

    # RSI 14
    df["rsi"] = RSIIndicator(close=close, window=14).rsi() if n >= 14 else None

    # MACD (12, 26, 9)
    if n >= 26:
        macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"]        = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"]   = macd_obj.macd_diff()
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = None

    # Bollinger Bands (20, 2)
    if n >= 20:
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"]    = bb.bollinger_hband()
        df["bb_lower"]    = bb.bollinger_lband()
        df["bb_mid"]      = bb.bollinger_mavg()
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / close
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    else:
        df["bb_upper"] = df["bb_lower"] = df["bb_mid"] = df["bb_width"] = df["bb_position"] = None

    # ATR 14
    df["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range() if n >= 14 else None

    # Stochastic (14, 3)
    if n >= 14:
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
    else:
        df["stoch_k"] = df["stoch_d"] = None

    # Williams %R 14
    df["williams_r"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r() if n >= 14 else None

    # Ichimoku Cloud (9, 26, 52)
    if n >= 52:
        ichi = IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
        df["ichi_conv"]  = ichi.ichimoku_conversion_line()   # Tenkan-sen
        df["ichi_base"]  = ichi.ichimoku_base_line()          # Kijun-sen
        df["ichi_a"]     = ichi.ichimoku_a()                  # Senkou Span A
        df["ichi_b"]     = ichi.ichimoku_b()                  # Senkou Span B
    else:
        df["ichi_conv"] = df["ichi_base"] = df["ichi_a"] = df["ichi_b"] = None

    # VWAP (rolling 20-bar proxy — true VWAP resets daily, this is a session approximation)
    if n >= 1:
        typical_price = (high + low + close) / 3
        df["vwap"] = (typical_price * volume).rolling(20, min_periods=1).sum() / \
                     (volume.rolling(20, min_periods=1).sum() + 1e-10)
    else:
        df["vwap"] = None

    # ── Round numeric values for clean JSON ──
    round4 = ["open", "high", "low", "close", "sma", "ema",
              "macd", "macd_signal", "macd_hist",
              "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_position",
              "atr", "stoch_k", "stoch_d", "williams_r",
              "ichi_conv", "ichi_base", "ichi_a", "ichi_b", "vwap"]
    round2 = ["rsi"]

    for col in round4:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].round(4)
    for col in round2:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].round(2)
    
    # ── Convert to list of dicts — replace NaN with None ──
    result_data = df.replace({np.nan: None}).to_dict(orient="records")
    
    logger.info(f"[PROCESSED] {symbol}/{period} — {len(result_data)} clean bars, interval={interval}")
    
    return result_data
