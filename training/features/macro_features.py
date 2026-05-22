"""
FinAI Macro Features — v1.2
=============================
Makroekonomik ve piyasa geneli feature'ları yfinance üzerinden çeker ve
hisse senedi verisiyle birleştirir.

Neden makro feature'lar?
- Hisse senetleri izole hareket etmez; Fed kararları, enflasyon, dolar endeksi
  ve piyasa korkusu (VIX) tüm varlıkları etkiler.
- Modelin "piyasa bağlamını" görmesi directional accuracy'yi artırır.
- Özellikle kriz dönemlerinde (2020, 2022) makro sinyaller teknik indikatörlerden
  çok daha erken uyarı verir.

Veri kaynakları (tümü yfinance — ücretsiz, API key gerektirmez):
┌─────────────────────────────────────────────────────────────────┐
│ Ticker    │ Ne                    │ Neden önemli               │
├─────────────────────────────────────────────────────────────────┤
│ ^VIX      │ CBOE Volatility Index │ Piyasa korku göstergesi    │
│ ^GSPC     │ S&P 500               │ Genel piyasa yönü          │
│ ^TNX      │ 10Y Treasury Yield    │ Risk-free rate, discount   │
│ ^IRX      │ 3M Treasury Yield     │ Kısa vade faiz             │
│ DX-Y.NYB  │ US Dollar Index (DXY) │ Dolar gücü                 │
│ GC=F      │ Gold Futures          │ Risk-off / enflasyon hedge │
│ CL=F      │ Crude Oil Futures     │ Enerji maliyeti, enflasyon │
│ ^IXIC     │ NASDAQ Composite      │ Tech sektör yönü           │
│ ^RUT      │ Russell 2000          │ Small-cap risk iştahı      │
│ HYG       │ High Yield Bond ETF   │ Kredi riski iştahı         │
│ TLT       │ 20Y Treasury ETF      │ Uzun vade tahvil           │
└─────────────────────────────────────────────────────────────────┘

Kullanım:
    from training.macro_features import fetch_macro_data, merge_macro_features
    
    macro_df = fetch_macro_data(period="5y")
    featured_df = merge_macro_features(stock_df, macro_df)
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Makro veri kaynakları
# ============================================================

MACRO_TICKERS = {
    # Volatilite & Korku
    "^VIX": "vix",                  # CBOE VIX — piyasa korku endeksi
    # Piyasa endeksleri
    "^GSPC": "sp500",               # S&P 500
    "^IXIC": "nasdaq",              # NASDAQ Composite
    "^RUT": "russell2000",          # Russell 2000 (small-cap)
    # Faiz & Tahvil
    "^TNX": "yield_10y",            # 10 yıllık ABD tahvil faizi
    "^IRX": "yield_3m",             # 3 aylık ABD tahvil faizi
    "TLT": "tlt",                   # 20Y Treasury Bond ETF
    "HYG": "hyg",                   # High Yield Corporate Bond ETF
    # Döviz
    "DX-Y.NYB": "dxy",              # US Dollar Index
    # Emtia
    "GC=F": "gold",                 # Altın vadeli
    "CL=F": "oil",                  # Ham petrol vadeli
}

# Sadece close fiyatı alınacak ticker'lar
MACRO_CLOSE_ONLY = set(MACRO_TICKERS.keys())


def fetch_macro_data(
    period: str = "5y",
    interval: str = "1d",
    delay: float = 0.3,
) -> pd.DataFrame:
    """
    Tüm makro ticker'ları çeker ve tek bir DataFrame'de birleştirir.
    
    Returns:
        DataFrame indexed by unix timestamp (time column),
        columns: vix, sp500, nasdaq, russell2000, yield_10y, yield_3m,
                 tlt, hyg, dxy, gold, oil
    """
    logger.info(f"Fetching macro data ({len(MACRO_TICKERS)} tickers)...")
    
    all_series = {}
    
    for ticker, col_name in MACRO_TICKERS.items():
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"  No data for {ticker} ({col_name})")
                continue
            
            df = df.reset_index()
            
            # Zaman sütununu normalize et
            time_col = df.get("Date", df.get("Datetime"))
            if time_col is None:
                logger.warning(f"  No time column for {ticker}")
                continue
            
            if hasattr(time_col.dt, "tz") and time_col.dt.tz is not None:
                time_col = time_col.dt.tz_convert("UTC").dt.tz_localize(None)
            
            timestamps = time_col.astype("int64") // 10**9
            close = pd.to_numeric(df["Close"], errors="coerce")
            
            series = pd.Series(close.values, index=timestamps.values, name=col_name)
            series = series.dropna()
            all_series[col_name] = series
            
            logger.info(f"  ✓ {ticker:12s} ({col_name:12s}): {len(series)} bars")
            time.sleep(delay)
            
        except Exception as e:
            logger.error(f"  ✗ {ticker} ({col_name}): {e}")
    
    if not all_series:
        logger.error("Hiç makro veri çekilemedi!")
        return pd.DataFrame()
    
    # Tüm serileri birleştir — ortak zaman ekseninde
    macro_df = pd.DataFrame(all_series)
    macro_df.index.name = "time"
    macro_df = macro_df.reset_index()
    macro_df = macro_df.sort_values("time").reset_index(drop=True)
    
    # Forward fill — hafta sonu / tatil günleri için
    macro_df = macro_df.set_index("time")
    macro_df = macro_df.ffill().bfill()
    macro_df = macro_df.reset_index()
    
    logger.info(f"Macro data: {len(macro_df)} rows, {len(macro_df.columns)-1} indicators")
    return macro_df


def build_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham makro fiyatlarından türetilmiş feature'lar oluşturur.
    
    Tüm feature'lar normalize edilmiş (ölçek bağımsız):
    - Return'ler: pct_change veya log-return
    - Seviyeler: z-score veya normalize edilmiş
    - Spread'ler: fark (zaten ölçek bağımsız)
    
    Returns:
        DataFrame with derived macro features
    """
    df = macro_df.copy()
    result = pd.DataFrame({"time": df["time"]})
    
    # ── VIX ──
    if "vix" in df.columns:
        result["vix_level"] = df["vix"]                                    # Seviye (0-80+)
        result["vix_change_1d"] = df["vix"].pct_change(1)                  # Günlük değişim
        result["vix_change_5d"] = df["vix"].pct_change(5)                  # Haftalık değişim
        result["vix_zscore_20"] = (
            (df["vix"] - df["vix"].rolling(20).mean()) /
            (df["vix"].rolling(20).std() + 1e-10)
        )
        result["vix_high_regime"] = (df["vix"] > 25).astype(int)           # Yüksek korku rejimi
        result["vix_extreme_regime"] = (df["vix"] > 40).astype(int)        # Kriz rejimi
    
    # ── S&P 500 ──
    if "sp500" in df.columns:
        result["sp500_return_1d"] = np.log(df["sp500"] / df["sp500"].shift(1))
        result["sp500_return_5d"] = np.log(df["sp500"] / df["sp500"].shift(5))
        result["sp500_return_20d"] = np.log(df["sp500"] / df["sp500"].shift(20))
        result["sp500_zscore_50"] = (
            (df["sp500"] - df["sp500"].rolling(50).mean()) /
            (df["sp500"].rolling(50).std() + 1e-10)
        )
        # Trend rejimi: 200 günlük MA üstünde mi?
        sp500_ma200 = df["sp500"].rolling(200).mean()
        result["sp500_bull_regime"] = (df["sp500"] > sp500_ma200).astype(int)
    
    # ── NASDAQ ──
    if "nasdaq" in df.columns:
        result["nasdaq_return_1d"] = np.log(df["nasdaq"] / df["nasdaq"].shift(1))
        result["nasdaq_return_5d"] = np.log(df["nasdaq"] / df["nasdaq"].shift(5))
    
    # ── Russell 2000 (risk iştahı göstergesi) ──
    if "russell2000" in df.columns:
        result["rut_return_1d"] = np.log(df["russell2000"] / df["russell2000"].shift(1))
        # Small-cap vs large-cap spread (risk iştahı)
        if "sp500" in df.columns:
            result["small_large_spread"] = (
                np.log(df["russell2000"] / df["russell2000"].shift(1)) -
                np.log(df["sp500"] / df["sp500"].shift(1))
            )
    
    # ── Faiz Oranları ──
    if "yield_10y" in df.columns:
        result["yield_10y"] = df["yield_10y"]                              # Seviye (%)
        result["yield_10y_change_1d"] = df["yield_10y"].diff(1)            # Günlük değişim (bps)
        result["yield_10y_change_5d"] = df["yield_10y"].diff(5)
        result["yield_10y_zscore_60"] = (
            (df["yield_10y"] - df["yield_10y"].rolling(60).mean()) /
            (df["yield_10y"].rolling(60).std() + 1e-10)
        )
    
    if "yield_3m" in df.columns:
        result["yield_3m"] = df["yield_3m"]
        result["yield_3m_change_1d"] = df["yield_3m"].diff(1)
    
    # ── Yield Curve (10Y - 3M spread) — resesyon göstergesi ──
    if "yield_10y" in df.columns and "yield_3m" in df.columns:
        result["yield_curve_spread"] = df["yield_10y"] - df["yield_3m"]
        result["yield_curve_inverted"] = (result["yield_curve_spread"] < 0).astype(int)
    
    # ── TLT (Uzun vade tahvil ETF) ──
    if "tlt" in df.columns:
        result["tlt_return_1d"] = np.log(df["tlt"] / df["tlt"].shift(1))
        result["tlt_return_5d"] = np.log(df["tlt"] / df["tlt"].shift(5))
    
    # ── HYG (High Yield — kredi riski) ──
    if "hyg" in df.columns:
        result["hyg_return_1d"] = np.log(df["hyg"] / df["hyg"].shift(1))
        # HYG vs TLT spread — kredi riski iştahı
        if "tlt" in df.columns:
            result["credit_risk_spread"] = (
                np.log(df["hyg"] / df["hyg"].shift(1)) -
                np.log(df["tlt"] / df["tlt"].shift(1))
            )
    
    # ── DXY (Dolar Endeksi) ──
    if "dxy" in df.columns:
        result["dxy_return_1d"] = np.log(df["dxy"] / df["dxy"].shift(1))
        result["dxy_return_5d"] = np.log(df["dxy"] / df["dxy"].shift(5))
        result["dxy_zscore_20"] = (
            (df["dxy"] - df["dxy"].rolling(20).mean()) /
            (df["dxy"].rolling(20).std() + 1e-10)
        )
        result["dxy_strong"] = (result["dxy_zscore_20"] > 1.0).astype(int)
    
    # ── Altın ──
    if "gold" in df.columns:
        result["gold_return_1d"] = np.log(df["gold"] / df["gold"].shift(1))
        result["gold_return_5d"] = np.log(df["gold"] / df["gold"].shift(5))
        result["gold_zscore_20"] = (
            (df["gold"] - df["gold"].rolling(20).mean()) /
            (df["gold"].rolling(20).std() + 1e-10)
        )
    
    # ── Ham Petrol ──
    if "oil" in df.columns:
        result["oil_return_1d"] = np.log(df["oil"] / df["oil"].shift(1))
        result["oil_return_5d"] = np.log(df["oil"] / df["oil"].shift(5))
        result["oil_zscore_20"] = (
            (df["oil"] - df["oil"].rolling(20).mean()) /
            (df["oil"].rolling(20).std() + 1e-10)
        )
    
    # ── Cross-asset momentum (risk-on / risk-off) ──
    # Risk-on: hisse yukarı + VIX aşağı + HYG yukarı
    # Risk-off: hisse aşağı + VIX yukarı + altın yukarı
    if all(c in result.columns for c in ["sp500_return_1d", "vix_change_1d"]):
        result["risk_on_signal"] = (
            (result["sp500_return_1d"] > 0).astype(int) +
            (result["vix_change_1d"] < 0).astype(int)
        ) / 2  # 0, 0.5, veya 1
    
    # ── Lag features (makro gecikmeli sinyaller) ──
    macro_cols = [c for c in result.columns if c != "time"]
    for col in ["vix_level", "yield_10y", "sp500_return_1d", "dxy_return_1d"]:
        if col in result.columns:
            result[f"{col}_lag_1"] = result[col].shift(1)
            result[f"{col}_lag_5"] = result[col].shift(5)
    
    # Infinity ve NaN temizliği
    result = result.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Built {len(result.columns)-1} macro features")
    return result


def merge_macro_features(
    stock_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    time_col: str = "time",
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Hisse senedi DataFrame'ine makro feature'ları birleştirir.
    
    Makro veriler günlük frekansta, hisse verileriyle aynı frekansta.
    Eksik günler (tatil, hafta sonu) forward-fill ile doldurulur.
    
    Args:
        stock_df: Hisse senedi DataFrame'i (time sütunu unix timestamp)
        macro_df: fetch_macro_data() çıktısı
        time_col: Zaman sütunu adı
        fill_method: Eksik değer doldurma yöntemi ('ffill' veya 'nearest')
    
    Returns:
        stock_df + makro feature sütunları
    """
    if macro_df.empty:
        logger.warning("Makro veri boş, birleştirme atlanıyor")
        return stock_df
    
    # Makro feature'ları hesapla
    macro_features = build_macro_features(macro_df)
    
    # Makro sütun adları (time hariç)
    macro_cols = [c for c in macro_features.columns if c != time_col]
    
    logger.info(f"Merging {len(macro_cols)} macro features into stock data...")
    
    # Merge: stock_df'in her satırı için en yakın makro değeri bul
    # as_of join: hisse tarihine <= olan en son makro değeri al
    # FIX v2.0: dtype uyumsuzluğunu önlemek için int64'e cast et
    stock_times = stock_df[time_col].values.astype(np.int64)
    
    # Her hisse zaman damgası için makro index'i bul (searchsorted)
    macro_features_sorted = macro_features.sort_values(time_col).reset_index(drop=True)
    macro_times_sorted = macro_features_sorted[time_col].values.astype(np.int64)
    
    # searchsorted: her stock_time için en yakın önceki makro index
    indices = np.searchsorted(macro_times_sorted, stock_times, side="right") - 1
    indices = np.clip(indices, 0, len(macro_features_sorted) - 1)
    
    # Makro değerleri al
    macro_values = macro_features_sorted[macro_cols].values[indices]
    macro_aligned = pd.DataFrame(macro_values, columns=macro_cols, index=stock_df.index)
    
    # NaN kontrolü — merge başarısız ise uyar
    nan_ratio = macro_aligned.isna().mean().mean()
    if nan_ratio > 0.5:
        logger.warning(
            f"Macro merge: {nan_ratio:.0%} NaN detected. "
            f"stock_time range: [{stock_times.min()}, {stock_times.max()}], "
            f"macro_time range: [{macro_times_sorted.min()}, {macro_times_sorted.max()}]"
        )
    
    # Forward-fill kalan NaN'ları
    macro_aligned = macro_aligned.ffill().bfill()
    
    result = pd.concat([stock_df.reset_index(drop=True), macro_aligned.reset_index(drop=True)], axis=1)
    
    logger.info(f"Merged: {len(result)} rows, {len(result.columns)} columns total")
    return result


def get_macro_feature_columns() -> List[str]:
    """
    Makro feature sütun adlarının listesini döner.
    get_feature_columns() ile birlikte kullanılır.
    """
    # build_macro_features'ın ürettiği sütunlar (time hariç)
    return [
        # VIX
        "vix_level", "vix_change_1d", "vix_change_5d", "vix_zscore_20",
        "vix_high_regime", "vix_extreme_regime",
        # S&P 500
        "sp500_return_1d", "sp500_return_5d", "sp500_return_20d",
        "sp500_zscore_50", "sp500_bull_regime",
        # NASDAQ
        "nasdaq_return_1d", "nasdaq_return_5d",
        # Russell 2000
        "rut_return_1d", "small_large_spread",
        # Faiz
        "yield_10y", "yield_10y_change_1d", "yield_10y_change_5d", "yield_10y_zscore_60",
        "yield_3m", "yield_3m_change_1d",
        "yield_curve_spread", "yield_curve_inverted",
        # Tahvil ETF
        "tlt_return_1d", "tlt_return_5d",
        "hyg_return_1d", "credit_risk_spread",
        # Döviz
        "dxy_return_1d", "dxy_return_5d", "dxy_zscore_20", "dxy_strong",
        # Emtia
        "gold_return_1d", "gold_return_5d", "gold_zscore_20",
        "oil_return_1d", "oil_return_5d", "oil_zscore_20",
        # Cross-asset
        "risk_on_signal",
        # Lag'lar
        "vix_level_lag_1", "vix_level_lag_5",
        "yield_10y_lag_1", "yield_10y_lag_5",
        "sp500_return_1d_lag_1", "sp500_return_1d_lag_5",
        "dxy_return_1d_lag_1", "dxy_return_1d_lag_5",
    ]


def save_macro_data(macro_df: pd.DataFrame, data_dir: str) -> str:
    """Makro veriyi parquet olarak kaydeder."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "macro_1d.parquet")
    macro_df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"Saved macro data: {path} ({size_mb:.1f} MB)")
    return path


def load_macro_data(data_dir: str) -> pd.DataFrame:
    """Kaydedilmiş makro veriyi yükler."""
    path = os.path.join(data_dir, "macro_1d.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Macro data not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded macro data: {len(df)} rows")
    return df
