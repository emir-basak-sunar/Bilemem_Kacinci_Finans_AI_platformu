"""
Redis Cache Manager for FinAI Platform
Handles all caching: market data, AI predictions, rate limiting
"""
import redis
import json
import hashlib
import time
import logging
from datetime import datetime
from typing import Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Redis Connection (singleton)
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Get or create Redis connection with connection pooling."""
    global _redis_client
    if _redis_client is None:
        try:
            pool = redis.ConnectionPool(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                max_connections=20,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True
            )
            _redis_client = redis.Redis(connection_pool=pool)
            _redis_client.ping()
            logger.info("Redis connection established successfully")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            _redis_client = None
    return _redis_client


def is_market_open() -> bool:
    """Check if US stock market is currently open (Eastern Time)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    
    now_et = datetime.now(ZoneInfo("America/New_York"))
    hour, minute = now_et.hour, now_et.minute
    weekday = now_et.weekday()  # 0=Monday, 6=Sunday
    
    if weekday >= 5:  # Weekend
        return False
    
    # Regular trading hours: 9:30 AM - 4:00 PM ET
    market_open = (hour == 9 and minute >= 30) or (10 <= hour < 16)
    return market_open


def get_ttl_for_market_data(period: str) -> int:
    """
    Dynamic TTL based on data period and market state.
    
    Strategy:
    - Intraday data (1d, 5d, 1mo): Short TTL when market is open
    - Historical data (3mo+): Longer TTL since data changes less frequently
    - Weekend/after-hours: Much longer TTL since data won't change
    """
    market_open = is_market_open()
    
    ttl_map_open = {
        "1d": 30,        # 30 seconds - very fresh for day traders
        "5d": 60,        # 1 minute
        "1mo": 60,       # 1 minute
        "3mo": 1800,     # 30 minutes
        "6mo": 1800,     # 30 minutes
        "1y": 21600,     # 6 hours
        "2y": 21600,     # 6 hours
        "5y": 43200,     # 12 hours
    }
    
    ttl_map_closed = {
        "1d": 21600,     # 6 hours - data won't change
        "5d": 21600,     # 6 hours
        "1mo": 21600,    # 6 hours
        "3mo": 43200,    # 12 hours
        "6mo": 43200,    # 12 hours
        "1y": 86400,     # 24 hours
        "2y": 86400,     # 24 hours
        "5y": 86400,     # 24 hours
    }
    
    if market_open:
        return ttl_map_open.get(period, 300)   # Default: 5 minutes
    else:
        return ttl_map_closed.get(period, 21600)  # Default: 6 hours


def get_ttl_for_prediction(period: str) -> int:
    """TTL for AI predictions. Shorter since they depend on current data."""
    return 300  # 5 minutes — predictions are invalidated by data_hash anyway


# ============================================================
# Cache key builders
# ============================================================

def market_data_key(symbol: str, period: str) -> str:
    """Cache key for market data: market:data:{SYMBOL}:{period}"""
    return f"market:data:{symbol.upper()}:{period}"


def prediction_key(symbol: str, model_type: str, period: str, horizon: int, data_hash: str) -> str:
    """
    Cache key for AI predictions.
    Includes data_hash so predictions auto-invalidate when underlying data changes.
    """
    return f"ai:prediction:{symbol.upper()}:{model_type}:{period}:{horizon}:{data_hash}"


def compute_data_hash(market_data: list) -> str:
    """
    Compute a short hash of the last N prices.
    Used to invalidate prediction cache when market data changes.
    """
    if not market_data:
        return "empty"
    # Use last 10 close prices for hash
    last_closes = [str(round(d.get("close", 0), 2)) for d in market_data[-10:]]
    content = "|".join(last_closes)
    return hashlib.md5(content.encode()).hexdigest()[:8]


# ============================================================
# Core cache operations with stampede prevention
# ============================================================

def cache_get(key: str) -> Optional[Any]:
    """Get value from Redis cache. Returns None on miss or error."""
    r = get_redis()
    if r is None:
        return None
    try:
        data = r.get(key)
        if data:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(data)
        logger.debug(f"Cache MISS: {key}")
        return None
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.warning(f"Cache read error for {key}: {e}")
        return None


def cache_set(key: str, value: Any, ttl: int) -> bool:
    """Set value in Redis cache with TTL. Returns False on error."""
    r = get_redis()
    if r is None:
        return False
    try:
        r.setex(key, ttl, json.dumps(value))
        logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        return True
    except (redis.RedisError, TypeError) as e:
        logger.warning(f"Cache write error for {key}: {e}")
        return False


def cache_delete(key: str) -> bool:
    """Delete a cache key."""
    r = get_redis()
    if r is None:
        return False
    try:
        r.delete(key)
        return True
    except redis.RedisError:
        return False


def acquire_lock(key: str, timeout: int = 5) -> bool:
    """
    Distributed lock for cache stampede prevention.
    Only one process/thread can hold the lock for a given key.
    Uses Redis SET NX EX pattern.
    """
    r = get_redis()
    if r is None:
        return True  # If Redis is down, proceed without lock
    lock_key = f"lock:{key}"
    try:
        acquired = r.set(lock_key, "1", nx=True, ex=timeout)
        return bool(acquired)
    except redis.RedisError:
        return True


def release_lock(key: str) -> None:
    """Release distributed lock."""
    r = get_redis()
    if r is None:
        return
    lock_key = f"lock:{key}"
    try:
        r.delete(lock_key)
    except redis.RedisError:
        pass


# ============================================================
# Cache statistics (for monitoring)
# ============================================================

def get_cache_stats() -> dict:
    """Get cache hit/miss statistics."""
    r = get_redis()
    if r is None:
        return {"status": "disconnected"}
    try:
        info = r.info("stats")
        return {
            "status": "connected",
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": round(
                info.get("keyspace_hits", 0) / 
                max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100, 2
            ),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": r.info("memory").get("used_memory_human", "N/A"),
        }
    except redis.RedisError:
        return {"status": "error"}
