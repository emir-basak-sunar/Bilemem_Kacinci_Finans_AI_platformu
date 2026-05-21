"""
Market Regime Detection Service
================================
Detects the current market regime using a Gaussian HMM (Hidden Markov Model).

Regimes:
  0 → Bull       — positive returns, low-medium volatility
  1 → Bear       — negative returns, elevated volatility
  2 → Sideways   — near-zero returns, low volatility
  3 → High Vol   — extreme volatility regardless of direction

Features fed to HMM (all normalized):
  - 5-bar log return
  - 20-bar rolling volatility (std of log returns)
  - 5-bar volume ratio vs 20-bar average
  - RSI (14) normalized to [-1, 1]
  - ATR ratio (ATR / price) — relative volatility

Redis caching:
  - Regime result: 30 min (tied to market data freshness)
  - Fitted HMM model: 24h per symbol (re-fit daily)

Design decisions:
  - GaussianHMM with 4 states, full covariance
  - Minimum 60 bars required (otherwise returns "insufficient data")
  - States are labeled post-fit by their return/volatility signature
  - No training data stored — model is fit on-the-fly from live market data
  - Deterministic seed for reproducibility
"""
import logging
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Optional

from services.redis_cache import cache_get, cache_set, get_redis

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_STATES       = 4
MIN_BARS       = 60       # minimum bars needed for reliable HMM fit
TTL_REGIME     = 1800     # 30 min — regime result cache
TTL_MODEL      = 86400    # 24h  — fitted HMM model cache
RANDOM_SEED    = 42

# Regime labels (assigned after fitting based on return/vol signature)
REGIME_LABELS = {
    "bull":     {"label": "Bull Market",     "color": "#22c55e", "icon": "📈", "description": "Sustained uptrend with positive momentum"},
    "bear":     {"label": "Bear Market",     "color": "#ef4444", "icon": "📉", "description": "Sustained downtrend with negative momentum"},
    "sideways": {"label": "Sideways",        "color": "#f59e0b", "icon": "↔️",  "description": "Range-bound, low directional conviction"},
    "high_vol": {"label": "High Volatility", "color": "#a855f7", "icon": "⚡", "description": "Elevated volatility, uncertain direction"},
}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_regime(market_data: list, symbol: str, period: str = "1mo") -> dict:
    """
    Detect the current market regime from OHLCV + indicator data.

    Args:
        market_data: List of OHLCV dicts (from market_data service)
        symbol:      Stock/crypto symbol
        period:      History period (used for cache key)

    Returns:
        {
          current_regime: str,          # "bull" | "bear" | "sideways" | "high_vol"
          current_label:  str,          # Human-readable label
          color:          str,          # Hex color for UI
          icon:           str,          # Emoji icon
          description:    str,          # One-line description
          confidence:     float,        # 0-1, probability of current state
          regime_history: list[dict],   # Last N bars with regime label
          transition_probs: dict,       # P(next regime | current regime)
          stats: dict,                  # Current window stats
          bars_analyzed: int,
          _response_time_ms: float,
        }
    """
    t0 = time.time()

    if not market_data or len(market_data) < MIN_BARS:
        return {
            "error": f"Need at least {MIN_BARS} bars for regime detection (got {len(market_data) if market_data else 0})",
            "current_regime": "unknown",
            "current_label": "Insufficient Data",
            "color": "#71717a",
            "icon": "❓",
            "description": "Not enough historical data",
            "confidence": 0.0,
            "regime_history": [],
            "transition_probs": {},
            "stats": {},
            "bars_analyzed": len(market_data) if market_data else 0,
            "_response_time_ms": 0.0,
        }

    # ── Cache check ──
    data_hash = _data_hash(market_data)
    cache_key = f"regime:{symbol.upper()}:{period}:{data_hash}"
    cached = cache_get(cache_key)
    if cached is not None:
        cached["_cache"] = "hit"
        return cached

    try:
        result = _run_hmm(market_data, symbol, period, data_hash)
        result["_response_time_ms"] = round((time.time() - t0) * 1000, 1)
        result["_cache"] = "miss"
        cache_set(cache_key, result, TTL_REGIME)
        return result
    except Exception as e:
        logger.error(f"Regime detection failed for {symbol}: {e}", exc_info=True)
        return {
            "error": str(e),
            "current_regime": "unknown",
            "current_label": "Detection Failed",
            "color": "#71717a",
            "icon": "❓",
            "description": "Regime detection encountered an error",
            "confidence": 0.0,
            "regime_history": [],
            "transition_probs": {},
            "stats": {},
            "bars_analyzed": len(market_data),
            "_response_time_ms": round((time.time() - t0) * 1000, 1),
        }


# ── Core HMM logic ────────────────────────────────────────────────────────────

def _run_hmm(market_data: list, symbol: str, period: str, data_hash: str) -> dict:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    df = pd.DataFrame(market_data)
    df = df.sort_values("time").reset_index(drop=True)

    # ── Feature engineering ──
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float).replace(0, np.nan).fillna(method="ffill").fillna(1)

    # Log returns (5-bar)
    log_ret = np.log(close / close.shift(1)).fillna(0)
    ret5    = log_ret.rolling(5).sum().fillna(0)

    # Rolling volatility (20-bar std of log returns)
    vol20 = log_ret.rolling(20).std().fillna(log_ret.std())

    # Volume ratio (5-bar avg / 20-bar avg)
    vol_ratio = (volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, 1)).fillna(1)
    vol_ratio = vol_ratio.clip(0.1, 10)

    # RSI normalized to [-1, 1]
    rsi_raw = df.get("rsi", pd.Series(dtype=float))
    if rsi_raw is None or rsi_raw.isna().all():
        from ta.momentum import RSIIndicator
        rsi_raw = RSIIndicator(close=close, window=14).rsi()
    rsi_norm = ((rsi_raw.fillna(50) - 50) / 50).clip(-1, 1)

    # ATR ratio (relative volatility)
    atr_raw = df.get("atr", pd.Series(dtype=float))
    if atr_raw is None or atr_raw.isna().all():
        from ta.volatility import AverageTrueRange
        high = df["high"].astype(float)
        low  = df["low"].astype(float)
        atr_raw = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    atr_ratio = (atr_raw.fillna(atr_raw.mean()) / close.replace(0, np.nan).fillna(method="ffill")).fillna(0)

    # ── Build feature matrix ──
    features = pd.DataFrame({
        "ret5":      ret5,
        "vol20":     vol20,
        "vol_ratio": vol_ratio,
        "rsi_norm":  rsi_norm,
        "atr_ratio": atr_ratio,
    }).dropna()

    if len(features) < MIN_BARS:
        raise ValueError(f"Not enough clean feature rows: {len(features)}")

    X = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit or load cached HMM ──
    model = _get_or_fit_hmm(X_scaled, symbol, period, data_hash)

    # ── Decode hidden states ──
    log_prob, state_seq = model.decode(X_scaled, algorithm="viterbi")
    posteriors = model.predict_proba(X_scaled)  # shape: (T, N_STATES)

    # ── Label states by return/volatility signature ──
    state_labels = _label_states(model, scaler, features.columns.tolist())

    # ── Current regime ──
    current_state    = int(state_seq[-1])
    current_regime   = state_labels[current_state]
    current_conf     = float(posteriors[-1, current_state])
    regime_cfg       = REGIME_LABELS[current_regime]

    # ── Regime history (last 60 bars) ──
    history_len = min(60, len(state_seq))
    times       = df["time"].values[-len(state_seq):]
    closes      = close.values[-len(state_seq):]

    regime_history = []
    for i in range(-history_len, 0):
        s   = int(state_seq[i])
        lbl = state_labels[s]
        cfg = REGIME_LABELS[lbl]
        regime_history.append({
            "time":       int(times[i]),
            "close":      round(float(closes[i]), 4),
            "regime":     lbl,
            "label":      cfg["label"],
            "color":      cfg["color"],
            "confidence": round(float(posteriors[i, s]), 3),
        })

    # ── Transition probabilities from current state ──
    trans_row = model.transmat_[current_state]
    transition_probs = {
        state_labels[j]: round(float(trans_row[j]), 3)
        for j in range(N_STATES)
    }

    # ── Current window stats ──
    last_n = min(20, len(features))
    recent = features.iloc[-last_n:]
    stats = {
        "return_5bar":    round(float(recent["ret5"].iloc[-1]) * 100, 3),
        "volatility_20":  round(float(recent["vol20"].mean()) * 100, 4),
        "volume_ratio":   round(float(recent["vol_ratio"].iloc[-1]), 3),
        "rsi":            round(float((recent["rsi_norm"].iloc[-1] * 50) + 50), 1),
        "atr_pct":        round(float(recent["atr_ratio"].iloc[-1]) * 100, 3),
        "regime_changes": int(_count_regime_changes(state_seq[-20:])),
    }

    # ── Regime distribution over history ──
    unique, counts = np.unique(state_seq, return_counts=True)
    regime_dist = {}
    for s, c in zip(unique, counts):
        lbl = state_labels[int(s)]
        regime_dist[lbl] = round(float(c) / len(state_seq), 3)

    return {
        "current_regime":   current_regime,
        "current_label":    regime_cfg["label"],
        "color":            regime_cfg["color"],
        "icon":             regime_cfg["icon"],
        "description":      regime_cfg["description"],
        "confidence":       round(current_conf, 3),
        "regime_history":   regime_history,
        "transition_probs": transition_probs,
        "regime_distribution": regime_dist,
        "stats":            stats,
        "bars_analyzed":    len(state_seq),
        "n_states":         N_STATES,
    }


def _get_or_fit_hmm(X_scaled: np.ndarray, symbol: str, period: str, data_hash: str):
    """Load cached HMM model or fit a new one."""
    from hmmlearn.hmm import GaussianHMM

    model_key = f"regime:model:{symbol.upper()}:{period}:{data_hash}"

    # Try loading from Redis
    r = get_redis()
    if r:
        try:
            raw = r.get(model_key)
            if raw:
                model = pickle.loads(raw)
                logger.debug(f"HMM model cache HIT: {model_key}")
                return model
        except Exception:
            pass

    # Fit new model
    logger.info(f"Fitting HMM for {symbol}/{period} ({len(X_scaled)} bars)")
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    model.fit(X_scaled)

    # Cache fitted model
    if r:
        try:
            r.setex(model_key, TTL_MODEL, pickle.dumps(model))
        except Exception as e:
            logger.warning(f"Failed to cache HMM model: {e}")

    return model


def _label_states(model, scaler, feature_names: list) -> dict:
    """
    Assign regime labels to HMM states based on their mean feature values.

    Strategy:
    - Inverse-transform state means back to original feature space
    - Sort states by (return, volatility) to assign labels:
        * Highest return + low vol  → bull
        * Lowest return + high vol  → bear
        * Highest vol (any return)  → high_vol
        * Remaining                 → sideways
    """
    means_scaled = model.means_  # shape: (N_STATES, N_FEATURES)
    means_orig   = scaler.inverse_transform(means_scaled)

    ret_idx = feature_names.index("ret5")
    vol_idx = feature_names.index("vol20")

    state_info = []
    for i in range(N_STATES):
        state_info.append({
            "state": i,
            "ret":   float(means_orig[i, ret_idx]),
            "vol":   float(means_orig[i, vol_idx]),
        })

    # Sort by volatility descending
    by_vol = sorted(state_info, key=lambda x: x["vol"], reverse=True)

    labels = {}

    # Highest volatility → high_vol
    labels[by_vol[0]["state"]] = "high_vol"

    # Among remaining 3, sort by return
    remaining = sorted(by_vol[1:], key=lambda x: x["ret"], reverse=True)

    # Highest return → bull
    labels[remaining[0]["state"]] = "bull"
    # Lowest return → bear
    labels[remaining[2]["state"]] = "bear"
    # Middle → sideways
    labels[remaining[1]["state"]] = "sideways"

    return labels


def _count_regime_changes(state_seq: np.ndarray) -> int:
    """Count number of regime transitions in a sequence."""
    if len(state_seq) < 2:
        return 0
    return int(np.sum(np.diff(state_seq) != 0))


def _data_hash(market_data: list) -> str:
    """Short hash of last 10 close prices for cache invalidation."""
    last = [str(round(d.get("close", 0), 2)) for d in market_data[-10:]]
    return hashlib.md5("|".join(last).encode()).hexdigest()[:8]
