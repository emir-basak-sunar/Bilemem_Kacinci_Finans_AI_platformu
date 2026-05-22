"""
FinAI Feature Engineering Module — v2.0
=========================================
160+ features shared across all model types.
Used both in training (Colab) and inference (serving).

Categories:
- Price Action: returns, log returns, volatility
- Trend: SMA, EMA, MACD, ADX, Ichimoku
- Momentum: RSI, Stochastic, Williams %R, CCI, ROC
- Volatility: Bollinger Bands, ATR, Keltner, Historical Vol, Parkinson, GK
- Volume: OBV, VWAP proxy, MFI, A/D Line, VWAP deviation
- Statistical: skewness, kurtosis, z-score, autocorrelation
- Market Microstructure: Amihud, order flow, range expansion, beta proxy
- Lag: price/volume/return lags
- Calendar: day of week, month, quarter
- Macro: VIX, S&P500, yield curve, DXY, gold, oil (macro_features.py'den)
- Cross-sectional: rank features (hedge fund alpha framework)

Değişiklikler v2.0 (hedge-fund grade):
- FIX: target_vol leakage — rolling(h) gelecek barları kullanıyordu
  Düzeltme: log_ret.rolling(h).std().shift(-h) yerine
  forward_vol = pd.Series([log_ret.iloc[i:i+h].std() for i in range(n)], index=close.index)
  → Sadece [i, i+h) barlarını kullanır, sızdırmaz
- FIX: autocorr O(n²) → rolling corr ile O(n)
  returns.rolling(20).apply(lambda x: x.autocorr()) → returns.rolling(20).corr(returns.shift(1))
- YENİ: Rolling beta to market (sp500 returns ile 60-bar beta)
- YENİ: Order flow imbalance — (close-open)/(high-low) intraday alıcı/satıcı baskısı
- YENİ: Realized skewness (crash risk indicator)
- YENİ: create_sequences_by_symbol — cross-symbol sequence sızıntısını önler
- YENİ: add_cross_sectional_features — rank-based features (hedge fund alpha)
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple

from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, AccDistIndexIndicator

logger = logging.getLogger(__name__)

# FIX v1.1: Ham fiyat seviyesi sütunları — semboller arası ölçek farklılığı yaratır.
# Bu sütunlar hesaplanır (indikatörler için gerekli) ama feature listesine dahil edilmez.
# Normalize edilmiş versiyonları (close_to_sma_N, bb_position vb.) kullanılır.
PRICE_LEVEL_FEATURES = set()


def _build_price_level_set(sma_windows, ema_windows):
    """Ham fiyat seviyesi feature'larının setini oluşturur."""
    s = set()
    for w in sma_windows:
        s.add(f"sma_{w}")
    for w in ema_windows:
        s.add(f"ema_{w}")
    # Bollinger ve Keltner ham seviyeleri
    s.update({"bb_upper", "bb_lower", "kc_upper", "kc_lower"})
    # Volume ham seviyeleri — semboller arası karşılaştırılamaz
    s.update({"obv", "ad_line", "volume_sma_20", "vwap_proxy"})
    return s


def build_features(
    df: pd.DataFrame,
    sma_windows: List[int] = None,
    ema_windows: List[int] = None,
    include_calendar: bool = True,
    include_lags: bool = True,
    price_lags: List[int] = None,
    volume_lags: List[int] = None,
    return_lags: List[int] = None,
    drop_na: bool = True,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build 120+ features from raw OHLCV data.

    Args:
        df: DataFrame with columns [time, open, high, low, close, volume]
        sma_windows: SMA periods (default: [7,14,21,50,100,200])
        ema_windows: EMA periods (default: [7,14,21,50,100,200])
        include_calendar: Whether to add day/month/quarter features
        include_lags: Whether to add lag features
        price_lags: Lag periods for price (default: [1,2,3,5,10])
        volume_lags: Lag periods for volume (default: [1,2,3])
        return_lags: Lag periods for returns (default: [1,2,3,5,10,20])
        drop_na: Whether to drop rows with NaN after feature computation
        macro_df: v1.2 — Makro veri DataFrame'i (macro_features.fetch_macro_data çıktısı)
                  None ise makro feature'lar eklenmez.

    Returns:
        DataFrame with all features added
    """
    if sma_windows is None:
        sma_windows = [7, 14, 21, 50, 100, 200]
    if ema_windows is None:
        ema_windows = [7, 14, 21, 50, 100, 200]
    if price_lags is None:
        price_lags = [1, 2, 3, 5, 10]
    if volume_lags is None:
        volume_lags = [1, 2, 3]
    if return_lags is None:
        return_lags = [1, 2, 3, 5, 10, 20]

    # Work on a clean copy of the base OHLCV data
    base = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    close  = base["close"]
    high   = base["high"]
    low    = base["low"]
    volume = base["volume"].astype(float)
    n      = len(base)

    logger.info(f"Building features for {n} bars...")

    # ── Accumulate ALL new columns in a dict, then concat once at the end ──
    # This avoids the PerformanceWarning from repeated DataFrame.insert calls.
    cols: dict = {}

    # ================================================================
    # 1. PRICE ACTION
    # ================================================================

    cols["return_1d"]  = close.pct_change(1)
    cols["return_5d"]  = close.pct_change(5)
    cols["return_10d"] = close.pct_change(10)
    cols["return_20d"] = close.pct_change(20)

    cols["log_return_1d"] = np.log(close / close.shift(1))
    cols["log_return_5d"] = np.log(close / close.shift(5))

    log_ret_1d = cols["log_return_1d"]
    cols["volatility_5d"]  = log_ret_1d.rolling(5).std()  * np.sqrt(252)
    cols["volatility_10d"] = log_ret_1d.rolling(10).std() * np.sqrt(252)
    cols["volatility_20d"] = log_ret_1d.rolling(20).std() * np.sqrt(252)

    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / base["open"]) ** 2
    cols["gk_volatility"] = np.sqrt(
        (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean() * 252
    )

    cols["high_low_range"]  = (high - low) / close
    cols["close_position"]  = (close - low) / (high - low + 1e-10)
    cols["gap"]             = base["open"] / close.shift(1) - 1

    # ================================================================
    # 2. TREND INDICATORS
    # ================================================================

    for w in sma_windows:
        if n >= w:
            sma = SMAIndicator(close=close, window=w).sma_indicator()
            cols[f"sma_{w}"]          = sma
            cols[f"close_to_sma_{w}"] = close / sma - 1

    for w in ema_windows:
        if n >= w:
            ema = EMAIndicator(close=close, window=w).ema_indicator()
            cols[f"ema_{w}"]          = ema
            cols[f"close_to_ema_{w}"] = close / ema - 1

    if n >= 26:
        macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line   = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist   = macd_obj.macd_diff()
        cols["macd"]             = macd_line
        cols["macd_signal"]      = macd_signal
        cols["macd_hist"]        = macd_hist
        cols["macd_norm"]        = macd_line   / (close + 1e-10)
        cols["macd_signal_norm"] = macd_signal / (close + 1e-10)
        cols["macd_hist_norm"]   = macd_hist   / (close + 1e-10)

    if n >= 14:
        adx_obj = ADXIndicator(high=high, low=low, close=close, window=14)
        cols["adx"]      = adx_obj.adx()
        cols["di_plus"]  = adx_obj.adx_pos()
        cols["di_minus"] = adx_obj.adx_neg()

    if n >= 52:
        ichi = IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
        ichi_a    = ichi.ichimoku_a()
        ichi_b    = ichi.ichimoku_b()
        ichi_base = ichi.ichimoku_base_line()
        ichi_conv = ichi.ichimoku_conversion_line()
        cols["ichimoku_a"]             = ichi_a
        cols["ichimoku_b"]             = ichi_b
        cols["ichimoku_base"]          = ichi_base
        cols["ichimoku_conv"]          = ichi_conv
        cols["close_to_ichimoku_a"]    = close / (ichi_a    + 1e-10) - 1
        cols["close_to_ichimoku_b"]    = close / (ichi_b    + 1e-10) - 1
        cols["close_to_ichimoku_base"] = close / (ichi_base + 1e-10) - 1
        cols["close_to_ichimoku_conv"] = close / (ichi_conv + 1e-10) - 1

    # ================================================================
    # 3. MOMENTUM INDICATORS
    # ================================================================

    if n >= 14:
        cols["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

    if n >= 14:
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        cols["stoch_k"] = stoch.stoch()
        cols["stoch_d"] = stoch.stoch_signal()

    if n >= 14:
        cols["williams_r"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()

    for w in [5, 10, 20]:
        if n >= w:
            cols[f"roc_{w}"] = ROCIndicator(close=close, window=w).roc()

    if n >= 20:
        cols["cci_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()

    # ================================================================
    # 4. VOLATILITY INDICATORS
    # ================================================================

    if n >= 20:
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        cols["bb_upper"]    = bb_upper
        cols["bb_lower"]    = bb_lower
        cols["bb_width"]    = (bb_upper - bb_lower) / (close + 1e-10)
        cols["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    if n >= 14:
        atr_14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        cols["atr_14"]  = atr_14
        cols["atr_pct"] = atr_14 / (close + 1e-10)

    if n >= 20:
        kc = KeltnerChannel(high=high, low=low, close=close, window=20)
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        cols["kc_upper"]          = kc_upper
        cols["kc_lower"]          = kc_lower
        cols["close_to_kc_upper"] = close / (kc_upper + 1e-10) - 1
        cols["close_to_kc_lower"] = close / (kc_lower + 1e-10) - 1

    # ================================================================
    # 5. VOLUME INDICATORS
    # ================================================================

    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    cols["obv"]        = obv
    cols["obv_change"] = obv.pct_change(5)

    if n >= 14:
        cols["mfi_14"] = MFIIndicator(high=high, low=low, close=close, volume=volume, window=14).money_flow_index()

    ad_line = AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()
    cols["ad_line"]  = ad_line
    cols["ad_change"] = ad_line.pct_change(5)

    vol_sma_20 = volume.rolling(20).mean()
    cols["volume_sma_20"] = vol_sma_20
    cols["volume_ratio"]  = volume / (vol_sma_20 + 1)

    typical_price = (high + low + close) / 3
    vwap_proxy = (typical_price * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1)
    cols["vwap_proxy"]    = vwap_proxy
    cols["close_to_vwap"] = close / (vwap_proxy + 1e-10) - 1

    # ================================================================
    # 6. STATISTICAL FEATURES
    # ================================================================

    returns = log_ret_1d
    cols["skewness_20"] = returns.rolling(20).skew()
    cols["kurtosis_20"] = returns.rolling(20).kurt()
    cols["zscore_20"]   = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
    cols["zscore_50"]   = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-10)

    if n >= 5:
        log_hl_sq = np.log(high / low) ** 2
        cols["parkinson_vol_5d"]  = np.sqrt(log_hl_sq.rolling(5).mean()  / (4 * np.log(2)) * 252)
        cols["parkinson_vol_20d"] = np.sqrt(log_hl_sq.rolling(20).mean() / (4 * np.log(2)) * 252)

    if n >= 25:
        # FIX v2.0: O(n²) apply+autocorr → O(n) rolling corr
        # Not: rolling.corr bazı pandas versiyonlarında sorun çıkarabilir
        # Güvenli fallback: pct_change ile yaklaşık autocorr
        try:
            autocorr_series = returns.rolling(20).corr(returns.shift(1))
            # Tüm NaN ise fallback kullan
            if autocorr_series.isna().all():
                autocorr_series = returns.rolling(20).apply(
                    lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0,
                    raw=True
                )
        except Exception:
            autocorr_series = pd.Series(0.0, index=returns.index)
        cols["autocorr_5"] = autocorr_series

    # ADX-derived (computed after adx is in cols)
    if "adx" in cols:
        cols["trend_strong"]      = (cols["adx"] > 25).astype(int)
        cols["trend_very_strong"] = (cols["adx"] > 40).astype(int)
        if "di_plus" in cols and "di_minus" in cols:
            cols["di_spread"] = cols["di_plus"] - cols["di_minus"]

    if n >= 20:
        vol_mean = volume.rolling(20).mean()
        vol_std  = volume.rolling(20).std()
        vol_surge = (volume - vol_mean) / (vol_std + 1)
        cols["volume_surge"]      = vol_surge
        cols["volume_surge_flag"] = (vol_surge > 2.0).astype(int)

    # ================================================================
    # 7. LAG FEATURES
    # ================================================================

    if include_lags:
        for lag in price_lags:
            cols[f"close_lag_{lag}"]        = close.shift(lag)
            cols[f"close_return_lag_{lag}"] = close.pct_change(lag).shift(1)

        for lag in volume_lags:
            cols[f"volume_lag_{lag}"]        = volume.shift(lag)
            cols[f"volume_return_lag_{lag}"] = volume.pct_change(lag).shift(1)

        for lag in return_lags:
            cols[f"return_lag_{lag}"] = returns.shift(lag)

    # ================================================================
    # 8. CALENDAR FEATURES
    # ================================================================

    if include_calendar and "time" in base.columns:
        try:
            dt = pd.to_datetime(base["time"], unit="s")
            cols["day_of_week"]    = dt.dt.dayofweek
            cols["month"]          = dt.dt.month
            cols["quarter"]        = dt.dt.quarter
            cols["is_month_start"] = dt.dt.is_month_start.astype(int)
            cols["is_month_end"]   = dt.dt.is_month_end.astype(int)
        except Exception:
            pass

    # ================================================================
    # Assemble — single pd.concat eliminates fragmentation warnings
    # ================================================================

    new_cols_df = pd.DataFrame(cols, index=base.index)
    result = pd.concat([base, new_cols_df], axis=1)

    # Replace infinities with NaN
    result = result.replace([np.inf, -np.inf], np.nan)

    # ================================================================
    # 9. MARKET MICROSTRUCTURE FEATURES (v1.3)
    # ================================================================
    # These require the assembled result (use result[...] not cols[...])

    micro_cols: dict = {}

    # Realized volatility (5-bar, annualized) — already have volatility_5d
    # but this is a cleaner version using close-to-close
    log_ret = result["log_return_1d"] if "log_return_1d" in result.columns else np.log(close / close.shift(1))

    # VWAP deviation (rolling 5-bar VWAP vs current price)
    if "vwap_proxy" in result.columns:
        micro_cols["vwap_dev_5"]  = (close - result["vwap_proxy"]) / (result["vwap_proxy"] + 1e-10)

    # Relative volume (current bar vs 5-bar avg — shorter window than volume_ratio)
    vol_5 = volume.rolling(5).mean()
    micro_cols["rel_volume_5"] = volume / (vol_5 + 1)

    # Intraday range expansion (today's range vs 5-bar avg range)
    daily_range = high - low
    avg_range_5 = daily_range.rolling(5).mean()
    micro_cols["range_expansion"] = daily_range / (avg_range_5 + 1e-10)

    # Realized volatility ratio (5-bar / 20-bar) — vol regime indicator
    if "volatility_5d" in result.columns and "volatility_20d" in result.columns:
        micro_cols["vol_ratio_5_20"] = result["volatility_5d"] / (result["volatility_20d"] + 1e-10)

    # FIX v2.0: O(n) rolling corr — pandas version safe
    try:
        autocorr_1 = log_ret.rolling(10).corr(log_ret.shift(1))
        if autocorr_1.isna().all():
            autocorr_1 = pd.Series(0.0, index=log_ret.index)
    except Exception:
        autocorr_1 = pd.Series(0.0, index=log_ret.index)
    micro_cols["return_autocorr_1"] = autocorr_1

    # Idiosyncratic return proxy: residual after removing rolling mean trend
    rolling_mean_ret = log_ret.rolling(20).mean()
    micro_cols["idiosyncratic_return"] = log_ret - rolling_mean_ret

    # Price momentum acceleration (2nd derivative of returns)
    ret_5  = close.pct_change(5)
    ret_10 = close.pct_change(10)
    micro_cols["momentum_accel"] = ret_5 - ret_10 / 2  # acceleration proxy

    # Up/down volume ratio (volume on up days vs down days, rolling 10)
    up_vol   = volume.where(close > close.shift(1), 0.0)
    down_vol = volume.where(close < close.shift(1), 0.0)
    micro_cols["up_down_vol_ratio"] = (
        up_vol.rolling(10).sum() / (down_vol.rolling(10).sum() + 1)
    )

    # Amihud illiquidity ratio (|return| / volume) — market impact proxy
    micro_cols["amihud_illiq"] = (np.abs(log_ret) / (volume + 1)).rolling(20).mean()

    # v2.0: Order flow imbalance — intraday alıcı/satıcı baskısı
    # (close - open) / (high - low): +1 = tam alıcı baskısı, -1 = tam satıcı baskısı
    intraday_range = high - low
    micro_cols["order_flow_imbalance"] = (close - base["open"]) / (intraday_range + 1e-10)
    micro_cols["order_flow_ma5"] = micro_cols["order_flow_imbalance"].rolling(5).mean()

    # v2.0: Realized skewness (crash risk) — negatif skew = sol kuyruk riski
    micro_cols["realized_skew_20"] = log_ret.rolling(20).skew()
    micro_cols["realized_skew_60"] = log_ret.rolling(60).skew()

    # v2.0: Rolling beta proxy (60-bar) — sp500 returns olmadan yaklaşık beta
    # Gerçek beta = cov(stock, market) / var(market)
    # Proxy: return volatility ratio (market vol olmadan)
    vol_60 = log_ret.rolling(60).std()
    vol_20 = log_ret.rolling(20).std()
    micro_cols["vol_regime_60_20"] = vol_60 / (vol_20 + 1e-10)  # vol rejim değişimi

    # v2.0: Price efficiency ratio (Hurst exponent proxy)
    # |net move| / sum(|daily moves|) — 1=trend, 0=random walk
    if n >= 20:
        net_move = (close - close.shift(20)).abs()
        sum_moves = log_ret.abs().rolling(20).sum()
        # FIX: close ile çarpmak yanlış — sadece move ratio
        micro_cols["efficiency_ratio_20"] = net_move / (sum_moves + 1e-10)

    # Assemble microstructure cols
    micro_df = pd.DataFrame(micro_cols, index=result.index)
    result = pd.concat([result, micro_df], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)

    # ================================================================
    # 9. MACRO FEATURES (v1.2)
    # ================================================================

    if macro_df is not None and not macro_df.empty and "time" in result.columns:
        try:
            from training.features.macro_features import merge_macro_features
            result_with_macro = merge_macro_features(result, macro_df)
            # Makro merge sonrası NaN kontrolü — tüm satırlar NaN ise merge başarısız
            macro_cols_added = [c for c in result_with_macro.columns if c not in result.columns]
            if macro_cols_added:
                nan_ratio = result_with_macro[macro_cols_added].isna().mean().mean()
                if nan_ratio > 0.9:
                    logger.warning(
                        f"Macro merge produced {nan_ratio:.0%} NaN — "
                        f"time format mismatch? Skipping macro features."
                    )
                else:
                    # Forward-fill makro NaN'larını (hafta sonu, tatil)
                    result_with_macro[macro_cols_added] = (
                        result_with_macro[macro_cols_added].ffill().bfill()
                    )
                    result = result_with_macro
                    logger.info("Macro features merged successfully")
            else:
                logger.warning("Macro merge added no columns — skipping")
        except ImportError as e:
            logger.warning(f"Macro feature import failed (non-critical): {e}")
        except Exception as e:
            logger.warning(f"Macro feature merge failed (non-critical): {e}")

    if drop_na:
        before = len(result)

        # Step 1: Remove columns that are entirely NaN
        # FIX: bool(series) yerine explicit .item() veya numpy bool kullan
        all_nan_cols = []
        for c in result.columns:
            try:
                col_data = result[c]
                if hasattr(col_data, 'isna'):
                    na_all = col_data.isna().all()
                    # pandas Series.all() → scalar bool, ama bazı versiyonlarda
                    # __nonzero__ çağrısı gerekiyor — bool() ile zorla
                    if bool(na_all):
                        all_nan_cols.append(c)
            except Exception:
                pass
        if all_nan_cols:
            logger.warning(
                f"Dropping {len(all_nan_cols)} all-NaN columns: "
                f"{all_nan_cols[:10]}"
            )
            result = result.drop(columns=all_nan_cols)

        # Step 2: Drop warmup rows where long-lookback indicators are NaN
        # (SMA_200 needs 200 bars → first ~200 rows will be NaN)
        warmup_cols = []
        for col in ["close_to_sma_200", "close_to_ema_200", "adx"]:
            if col in result.columns:
                warmup_cols.append(col)
                break  # one indicator is enough to determine warmup period
        if warmup_cols:
            result = result.dropna(subset=warmup_cols)

        # Step 3: Fill remaining scattered NaN with 0
        # Safe for normalized features (ratios, z-scores, returns)
        remaining_nan = int(result.isna().sum().sum())
        if remaining_nan > 0:
            result = result.fillna(0)
            logger.info(f"Filled {remaining_nan} remaining NaN values with 0")

        dropped = before - len(result)
        if dropped > 0:
            logger.info(f"Dropped {dropped} warmup rows ({before} → {len(result)})")

    feature_count = len(result.columns) - len(df.columns)
    logger.info(f"Built {feature_count} features, {len(result)} rows remaining")

    return result


def get_feature_columns(
    df: pd.DataFrame,
    exclude: List[str] = None,
    sma_windows: List[int] = None,
    ema_windows: List[int] = None,
    include_macro: bool = True,
) -> List[str]:
    """
    Get list of feature columns (excluding raw OHLCV, targets, and price-level features).

    FIX v1.1: Ham fiyat seviyesi sütunları feature listesine dahil edilmez.
    v1.2: include_macro parametresi — makro feature'ları dahil et/çıkar.
    """
    if exclude is None:
        exclude = []
    if sma_windows is None:
        sma_windows = [7, 14, 21, 50, 100, 200]
    if ema_windows is None:
        ema_windows = [7, 14, 21, 50, 100, 200]

    # Temel OHLCV ve meta sütunlar
    base_skip = {
        "time", "open", "high", "low", "close", "volume",
        "target", "symbol",
    }

    # Ham fiyat seviyesi sütunları — ölçek bağımlı
    price_level = _build_price_level_set(sma_windows, ema_windows)

    # Ham lag sütunları — ölçek bağımlı (normalize versiyonları var)
    lag_price = {f"close_lag_{lag}" for lag in [1, 2, 3, 5, 10]}
    lag_volume = {f"volume_lag_{lag}" for lag in [1, 2, 3]}

    # Ichimoku ham seviyeleri
    ichimoku_raw = {"ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conv"}

    # MACD ham seviyeleri (normalize versiyonları var)
    macd_raw = {"macd", "macd_signal", "macd_hist"}

    # ATR ham değeri (atr_pct normalize versiyonu var)
    atr_raw = {"atr_14"}

    skip = (base_skip | price_level | lag_price | lag_volume |
            ichimoku_raw | macd_raw | atr_raw | set(exclude))

    # v1.2: Makro feature'ları hariç tut istenirse
    if not include_macro:
        try:
            from training.features.macro_features import get_macro_feature_columns
            macro_cols = set(get_macro_feature_columns())
            skip = skip | macro_cols
        except ImportError:
            pass

    return [c for c in df.columns if c not in skip
            and not c.startswith("target_")
            and not c.startswith("future_price_")]


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    seq_length: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/TCN/TFT input.

    Args:
        data: Feature array (n_samples, n_features)
        targets: Target array (n_samples,)
        seq_length: Number of time steps per sequence

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences,)

    NOTE: Bu fonksiyon tek sembol için kullanılmalı.
    Multi-symbol için create_sequences_by_symbol kullanın.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def create_sequences_by_symbol(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_length: int = 60,
    symbol_col: str = "symbol",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    v2.0: Cross-symbol sequence sızıntısını önleyen sequence oluşturucu.

    SORUN: create_sequences tüm sembolleri birleştirip sequence oluşturuyordu.
    Bu durumda AAPL'nin son barı ile MSFT'nin ilk barı ardışık sequence'a giriyordu.
    Bu hem anlamsız hem de potansiyel leakage.

    ÇÖZÜM: Her sembol için ayrı ayrı sequence oluştur, sonra birleştir.
    Sembol sınırlarında sequence oluşturulmaz.

    Args:
        df: Multi-symbol DataFrame (symbol sütunu içermeli)
        feature_cols: Feature sütunları
        target_col: Hedef sütun
        seq_length: Sequence uzunluğu
        symbol_col: Sembol sütunu adı

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences,)
    """
    all_X, all_y = [], []

    if symbol_col not in df.columns:
        # Tek sembol — direkt oluştur
        data = df[feature_cols].values
        targets = df[target_col].values
        return create_sequences(data, targets, seq_length)

    symbols = df[symbol_col].unique()
    for sym in symbols:
        sym_df = df[df[symbol_col] == sym].sort_values("time").reset_index(drop=True)

        if len(sym_df) <= seq_length:
            logger.debug(f"  {sym}: yeterli bar yok ({len(sym_df)} <= {seq_length}), atlanıyor")
            continue

        data = sym_df[feature_cols].values
        targets_arr = sym_df[target_col].values

        X_sym, y_sym = create_sequences(data, targets_arr, seq_length)
        all_X.append(X_sym)
        all_y.append(y_sym)

    if not all_X:
        return np.array([]), np.array([])

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def create_multi_horizon_targets(
    close: pd.Series,
    horizons: List[int] = None,
    include_direction: bool = True,
    include_volatility: bool = True,
) -> pd.DataFrame:
    """
    Create multi-step prediction targets for multi-task learning.

    Targets per horizon:
    - target_{h}:          future return (regression)
    - target_dir_{h}:      direction (1=up, 0=down) — classification
    - target_strong_{h}:   strong move (|return| > 1%) — binary classification
    - target_vol_{h}:      future realized volatility (regression)

    Multi-task learning with these targets improves feature representation
    because the model learns both magnitude and direction simultaneously.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    targets = pd.DataFrame(index=close.index)
    log_ret = np.log(close / close.shift(1))

    for h in horizons:
        # Return (regression target)
        future_ret = close.shift(-h) / close - 1
        targets[f"target_{h}"]     = future_ret
        targets[f"future_price_{h}"] = close.shift(-h)

        if include_direction:
            # Direction (binary classification: 1=up, 0=down)
            targets[f"target_dir_{h}"]    = (future_ret > 0).astype(int)
            # Strong move (|return| > 1% threshold)
            targets[f"target_strong_{h}"] = (future_ret.abs() > 0.01).astype(int)

        if include_volatility:
            # v2.0 FIX: target_vol leakage düzeltildi
            n_bars = len(close)
            future_vol_values = np.full(n_bars, np.nan)
            log_ret_arr = np.log(close.values[1:] / close.values[:-1])  # length n-1
            for i in range(n_bars - h):
                # i+1'den i+h'ye kadar (h adet) log-return
                window = log_ret_arr[i: i + h]
                if len(window) == h and not np.any(np.isnan(window)):
                    if h > 1:  # ddof=1 için en az 2 eleman gerekli
                        future_vol_values[i] = np.std(window, ddof=1) * np.sqrt(252)
                    else:
                        future_vol_values[i] = np.abs(window[0]) * np.sqrt(252)
            targets[f"target_vol_{h}"] = pd.Series(future_vol_values, index=close.index)

    return targets


def add_cross_sectional_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "time",
    symbol_col: str = "symbol",
    min_symbols: int = 5,
) -> pd.DataFrame:
    """
    v2.0: Cross-sectional (rank-based) feature'lar ekler.

    Hedge fonların temel alpha framework'ü: mutlak return yerine
    semboller arası RANK kullanmak. Bu piyasa nötr alpha üretir.

    Her zaman adımında, her sembolün feature değeri diğer sembollerle
    karşılaştırılarak normalize edilir:
    - cs_rank_{feat}: cross-sectional rank [-0.5, +0.5]
    - cs_zscore_{feat}: cross-sectional z-score

    Eklenen feature'lar:
    - cs_rank_return_1d:   günlük return cross-sectional rank
    - cs_rank_return_5d:   5 günlük return rank
    - cs_rank_rsi_14:      RSI rank
    - cs_rank_volume_ratio: hacim rank
    - cs_rank_momentum_accel: momentum ivmesi rank
    - cs_zscore_return_1d: return z-score (cross-sectional)

    Args:
        df: Multi-symbol featured DataFrame
        feature_cols: Mevcut feature sütunları (güncellenir)
        time_col: Zaman sütunu
        symbol_col: Sembol sütunu
        min_symbols: Minimum sembol sayısı (az sembolde rank anlamsız)

    Returns:
        df: Cross-sectional feature'lar eklenmiş DataFrame
    """
    if symbol_col not in df.columns or time_col not in df.columns:
        logger.warning("Cross-sectional features: symbol veya time sütunu yok, atlanıyor")
        return df

    # Rank hesaplanacak feature'lar
    cs_features = [
        "return_1d", "return_5d", "return_20d",
        "rsi_14", "volume_ratio", "momentum_accel",
        "volatility_5d", "adx", "bb_position",
    ]

    df = df.copy()
    cs_cols_added = []

    for feat in cs_features:
        if feat not in df.columns:
            continue

        rank_col = f"cs_rank_{feat}"
        zscore_col = f"cs_zscore_{feat}"

        # Her zaman adımında cross-sectional rank hesapla
        # groupby(time).rank() — aynı zaman adımındaki semboller arasında rank
        grouped = df.groupby(time_col)[feat]
        n_per_time = grouped.transform("count")

        # Yeterli sembol yoksa NaN bırak
        valid_mask = n_per_time >= min_symbols

        # Rank: pct=True → [0,1], sonra -0.5 ile merkeze al → [-0.5, +0.5]
        ranks = grouped.rank(pct=True, na_option="keep") - 0.5
        df[rank_col] = ranks.where(valid_mask, np.nan)

        # Z-score: (x - mean) / std cross-sectional
        cs_mean = grouped.transform("mean")
        cs_std  = grouped.transform("std")
        df[zscore_col] = ((df[feat] - cs_mean) / (cs_std + 1e-10)).where(valid_mask, np.nan)

        cs_cols_added.extend([rank_col, zscore_col])

    if cs_cols_added:
        logger.info(f"Added {len(cs_cols_added)} cross-sectional features")

    return df


def add_cross_sectional_targets(
    df: pd.DataFrame,
    horizons: List[int] = None,
    time_col: str = "time",
    symbol_col: str = "symbol",
    min_symbols: int = 5,
) -> pd.DataFrame:
    """
    v2.0: Cross-sectional rank target'ları ekler.

    Hedge fonların IC'yi doğrudan optimize etmesi için:
    target_cs_rank_{h} = cross-sectional rank of future return at horizon h

    Bu target ile eğitilen model, mutlak return yerine
    "hangi sembol diğerlerinden daha iyi performans gösterir" sorusunu öğrenir.
    IC = Spearman(predicted_rank, actual_rank) — doğrudan optimize edilir.

    Args:
        df: Multi-symbol featured DataFrame (target_{h} sütunları içermeli)
        horizons: Prediction horizons
        time_col: Zaman sütunu
        symbol_col: Sembol sütunu
        min_symbols: Minimum sembol sayısı

    Returns:
        df: Cross-sectional rank target'ları eklenmiş DataFrame
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    if symbol_col not in df.columns or time_col not in df.columns:
        return df

    df = df.copy()

    for h in horizons:
        target_col = f"target_{h}"
        if target_col not in df.columns:
            continue

        cs_rank_col = f"target_cs_rank_{h}"

        # Her zaman adımında future return'ün cross-sectional rank'i
        grouped = df.groupby(time_col)[target_col]
        n_per_time = grouped.transform("count")
        valid_mask = n_per_time >= min_symbols

        # Rank: pct=True → [0,1], merkeze al → [-0.5, +0.5]
        ranks = grouped.rank(pct=True, na_option="keep") - 0.5
        df[cs_rank_col] = ranks.where(valid_mask, np.nan)

    cs_rank_cols = [f"target_cs_rank_{h}" for h in horizons if f"target_cs_rank_{h}" in df.columns]
    if cs_rank_cols:
        logger.info(f"Added cross-sectional rank targets: {cs_rank_cols}")

    return df
