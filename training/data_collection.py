"""
FinAI Data Collection — v1.1
==============================
Fetch historical OHLCV data from Yahoo Finance for training.
Supports multi-symbol, multi-timeframe collection.

Değişiklikler v1.1:
- train_test_split_timeseries: sembol bazlı split eklendi (split_by_symbol)
  Önceki davranış: tüm semboller karışık DataFrame'e tek split uygulanıyordu.
  Bu, AAPL'nin 2024 verisinin train'de, MSFT'nin 2024 verisinin test'te olmasına
  yol açıyordu — gerçek bir zaman serisi split değildi.
  Yeni davranış: her sembol kendi içinde split edilir, sonra birleştirilir.
- fetch_symbol_data: timezone-aware datetime desteği iyileştirildi
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import time
from typing import List, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def fetch_symbol_data(
    symbol: str,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single symbol.

    Returns DataFrame with columns: [time, open, high, low, close, volume, symbol]
    """
    logger.info(f"Fetching {symbol} — period={period}, interval={interval}")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "time",
            "Datetime": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # FIX v1.1: timezone-aware datetime'ları da yakala
        if "time" in df.columns:
            time_col = df["time"]
            if pd.api.types.is_datetime64_any_dtype(time_col):
                # timezone-aware ise UTC'ye normalize et
                if hasattr(time_col.dt, "tz") and time_col.dt.tz is not None:
                    time_col = time_col.dt.tz_convert("UTC").dt.tz_localize(None)
                df["time"] = time_col.astype("int64") // 10**9
            elif pd.api.types.is_object_dtype(time_col):
                # string datetime — parse et
                df["time"] = pd.to_datetime(time_col, utc=True).astype("int64") // 10**9

        # Keep only needed columns — KeyError'dan korunmak için mevcut olanları seç
        base_cols = ["time", "open", "high", "low", "close", "volume"]
        available = [c for c in base_cols if c in df.columns]
        df = df[available].copy()
        df["symbol"] = symbol

        # Clean
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("time").reset_index(drop=True)

        logger.info(f"  → {symbol}: {len(df)} bars ({df['time'].min()} to {df['time'].max()})")
        return df

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_multi_symbol(
    symbols: List[str],
    period: str = "5y",
    interval: str = "1d",
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch data for multiple symbols with rate limiting.
    Returns combined DataFrame with 'symbol' column.
    """
    all_data = []

    for i, symbol in enumerate(symbols):
        df = fetch_symbol_data(symbol, period, interval)
        if not df.empty:
            all_data.append(df)

        # Rate limiting
        if i < len(symbols) - 1:
            time.sleep(delay)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(combined)} bars across {len(all_data)} symbols")
    return combined


def save_data(df: pd.DataFrame, filename: str, data_dir: str = None):
    """Save DataFrame to parquet file in data directory."""
    if data_dir is None:
        from training.config import DATA_DIR
        data_dir = DATA_DIR

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"Saved {len(df)} rows to {path} ({size_mb:.1f} MB)")
    return path


def load_data(filename: str, data_dir: str = None) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    if data_dir is None:
        from training.config import DATA_DIR
        data_dir = DATA_DIR

    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """
    Time-series aware train/val/test split (no shuffling).
    Tek sembol veya sembol sütunu olmayan DataFrame'ler için kullanılır.

    Returns dict with keys: train, val, test
    """
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    # val_start: train içinden son val_ratio kadarı
    val_start = int(test_start * (1 - val_ratio / (1 - test_ratio)))

    splits = {
        "train": df.iloc[:val_start].copy(),
        "val": df.iloc[val_start:test_start].copy(),
        "test": df.iloc[test_start:].copy(),
    }

    for name, split in splits.items():
        logger.info(f"  {name}: {len(split)} rows ({len(split)/n*100:.1f}%)")

    return splits


def split_by_symbol(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    time_col: str = "time",
    symbol_col: str = "symbol",
) -> Dict[str, pd.DataFrame]:
    """
    FIX v1.1: Multi-symbol DataFrame için sembol bazlı zaman serisi split.

    Her sembol kendi içinde kronolojik olarak split edilir, ardından
    train/val/test setleri birleştirilir.

    Bu yaklaşım şu hatayı önler:
    - Eski: tüm semboller karışık sıralanmış DataFrame'e tek split
      → AAPL 2024 train'de, MSFT 2024 test'te olabilir
    - Yeni: her sembol kendi zaman ekseninde split edilir
      → tüm sembollerin son %20'si test'te

    Args:
        df: 'symbol' ve 'time' sütunları olan DataFrame
        test_ratio: Test seti oranı
        val_ratio: Validation seti oranı (train içinden)
        time_col: Zaman sütunu adı
        symbol_col: Sembol sütunu adı

    Returns:
        dict with keys: train, val, test
    """
    if symbol_col not in df.columns:
        logger.warning(f"'{symbol_col}' sütunu bulunamadı, tek sembol split kullanılıyor")
        return train_test_split_timeseries(df, test_ratio, val_ratio)

    trains, vals, tests = [], [], []
    symbols = df[symbol_col].unique()

    for sym in symbols:
        sym_df = df[df[symbol_col] == sym].sort_values(time_col).reset_index(drop=True)

        if len(sym_df) < 50:
            logger.warning(f"  {sym}: yeterli veri yok ({len(sym_df)} satır), atlanıyor")
            continue

        n = len(sym_df)
        test_start = int(n * (1 - test_ratio))
        val_start = int(test_start * (1 - val_ratio / (1 - test_ratio)))

        trains.append(sym_df.iloc[:val_start])
        vals.append(sym_df.iloc[val_start:test_start])
        tests.append(sym_df.iloc[test_start:])

    result = {
        "train": pd.concat(trains, ignore_index=True) if trains else pd.DataFrame(),
        "val":   pd.concat(vals,   ignore_index=True) if vals   else pd.DataFrame(),
        "test":  pd.concat(tests,  ignore_index=True) if tests  else pd.DataFrame(),
    }

    # Boş split kontrolü
    for name, split in result.items():
        if split.empty:
            logger.warning(f"  split '{name}' is empty — check data size (min 50 rows per symbol)")

    total = len(df)
    for name, split in result.items():
        logger.info(f"  {name}: {len(split)} rows ({len(split)/total*100:.1f}%) — {len(symbols)} sembol")

    return result
