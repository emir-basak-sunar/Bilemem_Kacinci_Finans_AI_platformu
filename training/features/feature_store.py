"""
FinAI Feature Store — v1.0
============================
Lightweight feature store inspired by FEAST architecture.

Neden bu yaklaşım?
  Tam FEAST kurulumu (feast CLI, feature server, BigQuery/Snowflake offline store)
  bu projeye overkill. Ama FEAST'in en değerli kısmını — feature registry,
  versiyonlama, materialization, point-in-time serving — Redis + Parquet ile
  hafif bir şekilde implemente ediyoruz.

Mimari:
  ┌─────────────────────────────────────────────────────────┐
  │                    Feature Store                         │
  │                                                          │
  │  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
  │  │  Feature     │    │  Offline     │    │  Online   │  │
  │  │  Registry    │    │  Store       │    │  Store    │  │
  │  │  (JSON)      │    │  (Parquet)   │    │  (Redis)  │  │
  │  └──────┬───────┘    └──────┬───────┘    └─────┬─────┘  │
  │         │                   │                   │        │
  │         └───────────────────┴───────────────────┘        │
  │                         │                                 │
  │                  FeatureStore API                         │
  └─────────────────────────────────────────────────────────┘

Bileşenler:
  1. FeatureView      — feature grubunu tanımlar (hangi feature'lar, TTL, entity)
  2. FeatureRegistry  — tüm FeatureView'ları kaydeder ve yönetir
  3. OfflineStore     — Parquet tabanlı tarihsel feature storage
  4. OnlineStore      — Redis tabanlı düşük-latency serving
  5. FeatureStore     — hepsini birleştiren ana API

Kullanım:
  # Training
  store = FeatureStore()
  store.materialize("IBM", market_data, period="1y")

  # Inference
  features = store.get_online_features("IBM", feature_view="technical")

  # Model kontrolü
  store.get_feature_stats("IBM")
  store.list_feature_views()
"""
import os
import json
import logging
import hashlib
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.dirname(_THIS_DIR)  # Go up from features/ to training/
STORE_DIR    = os.path.join(_TRAINING_DIR, "feature_store_data")
OFFLINE_DIR  = os.path.join(STORE_DIR, "offline")
REGISTRY_DIR = os.path.join(STORE_DIR, "registry")

for _d in [STORE_DIR, OFFLINE_DIR, REGISTRY_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── Feature View Definition ───────────────────────────────────────────────────

@dataclass
class FeatureView:
    """
    Defines a group of related features.

    Attributes:
        name:        Unique name (e.g. "technical", "momentum", "volume")
        features:    List of feature column names
        entity:      Entity key (e.g. "symbol")
        ttl_seconds: How long online features are valid (default: 1h)
        description: Human-readable description
        version:     Schema version (increment when features change)
        tags:        Arbitrary metadata tags
    """
    name:        str
    features:    List[str]
    entity:      str = "symbol"
    ttl_seconds: int = 3600
    description: str = ""
    version:     int = 1
    tags:        Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureView":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Pre-defined Feature Views (matching feature_engineering.py) ───────────────

def get_default_feature_views() -> List[FeatureView]:
    """
    Pre-defined feature views matching feature_engineering.py output.
    These are the canonical feature groups used across all models.
    """
    return [
        FeatureView(
            name="price_action",
            description="Returns, volatility, price position",
            features=[
                "return_1d", "return_5d", "return_10d", "return_20d",
                "log_return_1d", "log_return_5d",
                "volatility_5d", "volatility_10d", "volatility_20d",
                "gk_volatility", "high_low_range", "close_position", "gap",
            ],
            ttl_seconds=300,
            tags={"category": "price", "model_types": "all"},
        ),
        FeatureView(
            name="trend",
            description="SMA/EMA crossovers, MACD, ADX, Ichimoku",
            features=[
                "close_to_sma_7", "close_to_sma_14", "close_to_sma_21",
                "close_to_sma_50", "close_to_sma_100", "close_to_sma_200",
                "close_to_ema_7", "close_to_ema_14", "close_to_ema_21",
                "close_to_ema_50", "close_to_ema_100", "close_to_ema_200",
                "macd_norm", "macd_signal_norm", "macd_hist_norm",
                "adx", "di_plus", "di_minus", "trend_strong", "trend_very_strong", "di_spread",
                "close_to_ichimoku_a", "close_to_ichimoku_b",
                "close_to_ichimoku_base", "close_to_ichimoku_conv",
            ],
            ttl_seconds=3600,
            tags={"category": "trend", "model_types": "tree,dl"},
        ),
        FeatureView(
            name="momentum",
            description="RSI, Stochastic, Williams %R, CCI, ROC",
            features=[
                "rsi_14", "stoch_k", "stoch_d", "williams_r",
                "roc_5", "roc_10", "roc_20", "cci_20",
            ],
            ttl_seconds=300,
            tags={"category": "momentum", "model_types": "all"},
        ),
        FeatureView(
            name="volatility",
            description="Bollinger Bands, ATR, Keltner, Parkinson",
            features=[
                "bb_width", "bb_position",
                "atr_pct",
                "close_to_kc_upper", "close_to_kc_lower",
                "parkinson_vol_5d", "parkinson_vol_20d",
            ],
            ttl_seconds=300,
            tags={"category": "volatility", "model_types": "all"},
        ),
        FeatureView(
            name="volume",
            description="OBV, MFI, A/D, VWAP, volume ratios",
            features=[
                "obv_change", "mfi_14", "ad_change",
                "volume_ratio", "close_to_vwap",
                "volume_surge", "volume_surge_flag",
            ],
            ttl_seconds=300,
            tags={"category": "volume", "model_types": "all"},
        ),
        FeatureView(
            name="statistical",
            description="Skewness, kurtosis, z-score, autocorrelation",
            features=[
                "skewness_20", "kurtosis_20",
                "zscore_20", "zscore_50",
                "autocorr_5",
            ],
            ttl_seconds=3600,
            tags={"category": "statistical", "model_types": "tree"},
        ),
        FeatureView(
            name="calendar",
            description="Day of week, month, quarter, month boundaries",
            features=[
                "day_of_week", "month", "quarter",
                "is_month_start", "is_month_end",
            ],
            ttl_seconds=86400,
            tags={"category": "calendar", "model_types": "tree"},
        ),
        FeatureView(
            name="lags",
            description="Lagged returns and volume changes",
            features=[
                "close_return_lag_1", "close_return_lag_2", "close_return_lag_3",
                "close_return_lag_5", "close_return_lag_10",
                "volume_return_lag_1", "volume_return_lag_2", "volume_return_lag_3",
                "return_lag_1", "return_lag_2", "return_lag_3",
                "return_lag_5", "return_lag_10", "return_lag_20",
            ],
            ttl_seconds=300,
            tags={"category": "lags", "model_types": "tree"},
        ),
        FeatureView(
            name="symbol_meta",
            description="Symbol encoding for multi-symbol models",
            features=["symbol_encoded"],
            ttl_seconds=86400,
            tags={"category": "meta", "model_types": "tree"},
        ),
    ]


# ── Feature Registry ──────────────────────────────────────────────────────────

class FeatureRegistry:
    """
    Manages FeatureView definitions.
    Persists to JSON so views survive restarts.
    """

    def __init__(self, registry_dir: str = REGISTRY_DIR):
        self.registry_dir = registry_dir
        self._views: Dict[str, FeatureView] = {}
        self._load()

    def register(self, view: FeatureView, overwrite: bool = True):
        """Register a FeatureView."""
        if view.name in self._views and not overwrite:
            raise ValueError(f"FeatureView '{view.name}' already registered. Use overwrite=True.")
        self._views[view.name] = view
        self._save()
        logger.info(f"Registered FeatureView: {view.name} ({len(view.features)} features)")

    def register_defaults(self):
        """Register all default feature views."""
        for view in get_default_feature_views():
            self.register(view, overwrite=True)
        logger.info(f"Registered {len(self._views)} default feature views")

    def get(self, name: str) -> Optional[FeatureView]:
        return self._views.get(name)

    def list_views(self) -> List[str]:
        return sorted(self._views.keys())

    def get_all_features(self) -> List[str]:
        """Get flat list of all features across all views."""
        all_feats = []
        for view in self._views.values():
            all_feats.extend(view.features)
        return list(dict.fromkeys(all_feats))  # deduplicate, preserve order

    def get_features_for_model(self, model_type: str) -> List[str]:
        """Get features relevant for a specific model type."""
        feats = []
        for view in self._views.values():
            model_types = view.tags.get("model_types", "all")
            if model_types == "all" or model_type in model_types.split(","):
                feats.extend(view.features)
        return list(dict.fromkeys(feats))

    def _save(self):
        path = os.path.join(self.registry_dir, "feature_views.json")
        data = {name: view.to_dict() for name, view in self._views.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        path = os.path.join(self.registry_dir, "feature_views.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self._views = {name: FeatureView.from_dict(v) for name, v in data.items()}
            logger.debug(f"Loaded {len(self._views)} feature views from registry")
        else:
            # Bootstrap with defaults
            for view in get_default_feature_views():
                self._views[view.name] = view
            self._save()


# ── Offline Store ─────────────────────────────────────────────────────────────

class OfflineStore:
    """
    Parquet-based offline feature storage.
    Stores materialized features per symbol + period.
    Supports point-in-time retrieval.
    """

    def __init__(self, offline_dir: str = OFFLINE_DIR):
        self.offline_dir = offline_dir

    def write(self, symbol: str, period: str, df: pd.DataFrame):
        """Write feature DataFrame to offline store."""
        sym_dir = os.path.join(self.offline_dir, symbol.upper())
        os.makedirs(sym_dir, exist_ok=True)
        path = os.path.join(sym_dir, f"{period}.parquet")
        df.to_parquet(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        logger.info(f"Offline store write: {symbol}/{period} → {len(df)} rows, {len(df.columns)} cols ({size_kb:.0f} KB)")
        return path

    def read(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Read feature DataFrame from offline store."""
        path = os.path.join(self.offline_dir, symbol.upper(), f"{period}.parquet")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        logger.debug(f"Offline store read: {symbol}/{period} → {len(df)} rows")
        return df

    def get_latest_row(self, symbol: str, period: str) -> Optional[pd.Series]:
        """Get the most recent feature row (for online serving)."""
        df = self.read(symbol, period)
        if df is None or df.empty:
            return None
        if "time" in df.columns:
            df = df.sort_values("time")
        return df.iloc[-1]

    def list_symbols(self) -> List[str]:
        if not os.path.exists(self.offline_dir):
            return []
        return [d for d in os.listdir(self.offline_dir)
                if os.path.isdir(os.path.join(self.offline_dir, d))]

    def get_stats(self, symbol: str, period: str) -> Optional[dict]:
        """Get statistics about stored features."""
        df = self.read(symbol, period)
        if df is None:
            return None
        numeric = df.select_dtypes(include=[np.number])
        return {
            "symbol": symbol,
            "period": period,
            "rows": len(df),
            "features": len(df.columns),
            "time_range": [
                int(df["time"].min()) if "time" in df.columns else None,
                int(df["time"].max()) if "time" in df.columns else None,
            ],
            "null_pct": round(float(df.isnull().mean().mean()) * 100, 2),
            "feature_stats": {
                col: {
                    "mean": round(float(numeric[col].mean()), 6),
                    "std":  round(float(numeric[col].std()), 6),
                    "min":  round(float(numeric[col].min()), 6),
                    "max":  round(float(numeric[col].max()), 6),
                }
                for col in list(numeric.columns)[:20]  # top 20 features
            },
        }


# ── Online Store ──────────────────────────────────────────────────────────────

class OnlineStore:
    """
    Redis-based online feature store for low-latency serving.
    Stores the latest feature vector per (symbol, feature_view).
    """

    def __init__(self):
        self._redis = None
        self._local_cache: Dict[str, Any] = {}  # fallback when Redis unavailable

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379, db=1,  # db=1 to separate from app cache
                            decode_responses=True, socket_timeout=1)
            r.ping()
            self._redis = r
            return r
        except Exception:
            return None

    def write(self, symbol: str, view_name: str, features: dict, ttl: int = 3600):
        """Write feature vector to online store."""
        key = f"fs:online:{symbol.upper()}:{view_name}"
        payload = json.dumps({
            "features": features,
            "written_at": datetime.now(timezone.utc).isoformat(),
        })
        r = self._get_redis()
        if r:
            try:
                r.setex(key, ttl, payload)
                return True
            except Exception as e:
                logger.warning(f"Redis write failed: {e}")
        # Fallback to local cache
        self._local_cache[key] = payload
        return False

    def read(self, symbol: str, view_name: str) -> Optional[dict]:
        """Read feature vector from online store."""
        key = f"fs:online:{symbol.upper()}:{view_name}"
        r = self._get_redis()
        raw = None
        if r:
            try:
                raw = r.get(key)
            except Exception:
                pass
        if raw is None:
            raw = self._local_cache.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def read_all_views(self, symbol: str) -> Dict[str, dict]:
        """Read all feature views for a symbol."""
        r = self._get_redis()
        result = {}
        pattern = f"fs:online:{symbol.upper()}:*"
        if r:
            try:
                keys = r.keys(pattern)
                for key in keys:
                    view_name = key.split(":")[-1]
                    raw = r.get(key)
                    if raw:
                        result[view_name] = json.loads(raw)
            except Exception:
                pass
        return result

    def delete(self, symbol: str, view_name: str = None):
        """Delete online features for a symbol (optionally specific view)."""
        r = self._get_redis()
        if view_name:
            key = f"fs:online:{symbol.upper()}:{view_name}"
            if r:
                r.delete(key)
            self._local_cache.pop(key, None)
        else:
            pattern = f"fs:online:{symbol.upper()}:*"
            if r:
                keys = r.keys(pattern)
                if keys:
                    r.delete(*keys)


# ── Main Feature Store API ────────────────────────────────────────────────────

class FeatureStore:
    """
    Main Feature Store API.

    Combines FeatureRegistry + OfflineStore + OnlineStore.

    Usage:
        store = FeatureStore()

        # Materialize features (offline + online)
        store.materialize("IBM", market_data, period="1y")

        # Get features for inference
        features = store.get_online_features("IBM")

        # Inspect
        store.get_feature_stats("IBM")
        store.list_feature_views()
    """

    def __init__(self):
        self.registry = FeatureRegistry()
        self.offline  = OfflineStore()
        self.online   = OnlineStore()

    # ── Materialization ───────────────────────────────────────────────────────

    def materialize(
        self,
        symbol: str,
        market_data: list,
        period: str = "1y",
        symbol_encoding: Optional[Dict[str, int]] = None,
    ) -> dict:
        """
        Compute features from raw market data and store in both offline and online stores.

        Args:
            symbol:          Stock/crypto symbol
            market_data:     List of OHLCV dicts (from market_data service)
            period:          History period label (for offline store key)
            symbol_encoding: Optional symbol→int mapping for symbol_encoded feature

        Returns:
            Summary dict with materialization stats
        """
        t0 = time.time()
        sym = symbol.upper()

        if not market_data or len(market_data) < 30:
            return {"error": f"Insufficient data: {len(market_data) if market_data else 0} bars"}

        # Build features
        from training.features.feature_engineering import build_features, get_feature_columns
        df = pd.DataFrame(market_data)

        # FIX: drop_na=False — sma_200 warmup drop'u tüm veriyi silebilir.
        # Özellikle period="1y" → ~252 bar → warmup sonrası ~52 bar kalır ama
        # bazı durumlarda 0 satır kalabilir. Bunun yerine NaN'ları 0 ile doldur.
        featured = build_features(df, drop_na=False)
        featured = featured.replace([float('inf'), float('-inf')], float('nan'))
        featured = featured.fillna(0)
        # close > 0 olan satırları tut (geçerli fiyat verisi)
        if "close" in featured.columns:
            featured = featured[featured["close"] > 0]

        # Add symbol encoding
        if symbol_encoding:
            code = symbol_encoding.get(sym, -1)
        else:
            try:
                from training.config import SYMBOL_ENCODING
                code = SYMBOL_ENCODING.get(sym, -1)
            except ImportError:
                code = -1
        featured["symbol_encoded"] = code

        # Get clean feature columns
        feature_cols = get_feature_columns(featured)
        feature_cols = [c for c in feature_cols if c in featured.columns]

        # Write to offline store
        self.offline.write(sym, period, featured[["time"] + feature_cols])

        # Write latest row to online store (per feature view)
        latest = featured.iloc[-1]
        views_written = 0
        for view in self.registry._views.values():
            available = [f for f in view.features if f in latest.index]
            if not available:
                continue
            feat_dict = {f: _safe_val(latest[f]) for f in available}
            self.online.write(sym, view.name, feat_dict, ttl=view.ttl_seconds)
            views_written += 1

        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(f"Materialized {sym}/{period}: {len(featured)} rows, {len(feature_cols)} features, {views_written} views ({elapsed}ms)")

        return {
            "symbol":        sym,
            "period":        period,
            "rows":          len(featured),
            "features":      len(feature_cols),
            "views_written": views_written,
            "elapsed_ms":    elapsed,
        }

    # ── Online Serving ────────────────────────────────────────────────────────

    def get_online_features(
        self,
        symbol: str,
        feature_views: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest feature vector from online store.

        Args:
            symbol:        Stock symbol
            feature_views: List of view names to fetch (None = all)

        Returns:
            Dict of {feature_name: value} or None if not materialized
        """
        sym = symbol.upper()
        all_views = self.online.read_all_views(sym)

        if not all_views:
            return None

        merged = {}
        for view_name, payload in all_views.items():
            if feature_views and view_name not in feature_views:
                continue
            merged.update(payload.get("features", {}))

        return merged if merged else None

    def get_online_feature_view(self, symbol: str, view_name: str) -> Optional[dict]:
        """Get a specific feature view from online store."""
        payload = self.online.read(symbol, view_name)
        if payload is None:
            return None
        return payload.get("features")

    # ── Inspection & Monitoring ───────────────────────────────────────────────

    def get_feature_stats(self, symbol: str, period: str = "1y") -> Optional[dict]:
        """Get statistics about stored features for a symbol."""
        return self.offline.get_stats(symbol, period)

    def list_feature_views(self) -> List[dict]:
        """List all registered feature views with metadata."""
        result = []
        for name in self.registry.list_views():
            view = self.registry.get(name)
            result.append({
                "name":        view.name,
                "description": view.description,
                "features":    len(view.features),
                "ttl_seconds": view.ttl_seconds,
                "tags":        view.tags,
                "version":     view.version,
            })
        return result

    def list_materialized_symbols(self) -> List[str]:
        """List all symbols that have been materialized."""
        return self.offline.list_symbols()

    def get_registry_summary(self) -> dict:
        """Get a summary of the entire feature store."""
        views = self.list_feature_views()
        symbols = self.list_materialized_symbols()
        total_features = len(self.registry.get_all_features())
        return {
            "feature_views":         len(views),
            "total_features":        total_features,
            "materialized_symbols":  len(symbols),
            "symbols":               symbols,
            "views":                 views,
        }

    def validate_features(self, symbol: str, period: str, model_feature_cols: List[str]) -> dict:
        """
        Validate that stored features match what a model expects.
        Useful for catching feature drift before inference.
        """
        df = self.offline.read(symbol, period)
        if df is None:
            return {"valid": False, "error": "No offline features found"}

        stored_cols = set(df.columns) - {"time"}
        expected_cols = set(model_feature_cols)

        missing  = expected_cols - stored_cols
        extra    = stored_cols - expected_cols
        matched  = expected_cols & stored_cols

        return {
            "valid":         len(missing) == 0,
            "matched":       len(matched),
            "missing":       sorted(missing),
            "extra":         sorted(extra),
            "match_rate":    round(len(matched) / max(len(expected_cols), 1), 3),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_val(v) -> Any:
    """Convert numpy/pandas scalars to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ── Singleton ─────────────────────────────────────────────────────────────────

_store: Optional[FeatureStore] = None

def get_feature_store() -> FeatureStore:
    """Get or create the global FeatureStore singleton."""
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store
