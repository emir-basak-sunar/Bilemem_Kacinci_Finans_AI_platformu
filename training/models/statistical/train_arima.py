"""
FinAI Statistical Models — ARIMA / SARIMA / SARIMAX
=====================================================
Per-symbol training. CPU only.

Çalıştırma:
    python -m training.models.statistical.train_arima
"""
import sys
import os
import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.config import CORE_SYMBOLS, CRYPTO_SYMBOLS, ARIMA_CFG, DATA_DIR
from training.data_collection import fetch_multi_symbol, save_data
from training.model_registry import ModelRegistry
from training.models.tree.train_tree_models import evaluate_predictions

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _price_to_log_returns(prices: np.ndarray) -> np.ndarray:
    return np.log(prices[1:] / prices[:-1])


def _log_returns_to_price(log_returns: np.ndarray, last_price: float) -> np.ndarray:
    prices = [last_price]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])


def train_arima_for_symbol(
    df_symbol: pd.DataFrame,
    symbol: str,
    exog_cols: List[str] = None,
    test_ratio: float = 0.2,
) -> Dict:
    """
    Train ARIMA / SARIMA / SARIMAX for a single symbol on log-returns.

    Returns dict with keys: arima, sarima, sarimax (each has model, metrics, path).
    """
    close = df_symbol["close"].values
    n = len(close)
    split_idx = int(n * (1 - test_ratio))

    log_returns = _price_to_log_returns(close)
    train_returns = log_returns[:split_idx - 1]
    test_returns  = log_returns[split_idx - 1:]
    test_close    = close[split_idx:]
    last_train_price = close[split_idx - 1]

    results = {}
    registry = ModelRegistry()
    version  = registry.get_next_version(symbol)

    # ── 1. Auto-ARIMA ──
    logger.info(f"  [{symbol}] Auto-ARIMA (log-returns)...")
    t0 = time.time()
    arima_model = None
    try:
        arima_model = pm.auto_arima(
            train_returns,
            start_p=1, start_q=1,
            max_p=ARIMA_CFG.max_p, d=0, max_d=2, max_q=ARIMA_CFG.max_q,
            seasonal=False,
            stepwise=ARIMA_CFG.stepwise,
            suppress_warnings=True, error_action="ignore", trace=False,
        )
        arima_fc = arima_model.predict(n_periods=len(test_returns))
        arima_price_fc = _log_returns_to_price(arima_fc, last_train_price)

        m = evaluate_predictions(test_returns, arima_fc)
        m["price_rmse"]   = round(float(np.sqrt(np.mean((test_close - arima_price_fc) ** 2))), 4)
        m["train_time_s"] = round(time.time() - t0, 1)
        m["order"]        = str(arima_model.order)
        m["trained_on"]   = "log_returns"

        try:
            m["aic"] = round(float(arima_model.aic), 2)
            m["bic"] = round(float(arima_model.bic), 2)
        except Exception:
            pass

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(arima_model.resid(), lags=[10], return_df=True)
            m["ljungbox_p10"]  = round(float(lb["lb_pvalue"].iloc[0]), 4)
            m["residuals_ok"]  = bool(m["ljungbox_p10"] > 0.05)
        except Exception:
            pass

        path = registry.save_model(arima_model, symbol, "arima", version, format="pickle")
        results["arima"] = {"model": arima_model, "metrics": m, "path": path}
        logger.info(f"    ARIMA{arima_model.order}: sMAPE={m.get('smape', float('nan')):.2f}%, "
                    f"RMSE={m['rmse']:.6f}, price_RMSE={m['price_rmse']:.4f} ({m['train_time_s']}s)")
    except Exception as e:
        logger.error(f"    ARIMA failed for {symbol}: {e}")
        results["arima"] = {"error": str(e)}

    # ── 2. SARIMA (m=5 haftalık) ──
    logger.info(f"  [{symbol}] SARIMA (m=5)...")
    t0 = time.time()
    try:
        sarima_model = pm.auto_arima(
            train_returns,
            start_p=1, start_q=1,
            max_p=ARIMA_CFG.max_p, d=0, max_d=2, max_q=ARIMA_CFG.max_q,
            seasonal=True, m=5,
            start_P=0, start_Q=0, max_P=1, max_Q=1,
            stepwise=True, suppress_warnings=True, error_action="ignore",
        )
        sarima_fc = sarima_model.predict(n_periods=len(test_returns))
        sarima_price_fc = _log_returns_to_price(sarima_fc, last_train_price)

        m = evaluate_predictions(test_returns, sarima_fc)
        m["price_rmse"]      = round(float(np.sqrt(np.mean((test_close - sarima_price_fc) ** 2))), 4)
        m["train_time_s"]    = round(time.time() - t0, 1)
        m["order"]           = str(sarima_model.order)
        m["seasonal_order"]  = str(sarima_model.seasonal_order)
        m["trained_on"]      = "log_returns"

        path = registry.save_model(sarima_model, symbol, "sarima", version, format="pickle")
        results["sarima"] = {"model": sarima_model, "metrics": m, "path": path}
        logger.info(f"    SARIMA{sarima_model.order}x{sarima_model.seasonal_order}: "
                    f"sMAPE={m.get('smape', float('nan')):.2f}% ({m['train_time_s']}s)")
    except Exception as e:
        logger.error(f"    SARIMA failed for {symbol}: {e}")
        results["sarima"] = {"error": str(e)}

    # ── 3. SARIMAX (exog variables) ──
    if exog_cols and all(c in df_symbol.columns for c in exog_cols):
        logger.info(f"  [{symbol}] SARIMAX ({len(exog_cols)} exog)...")
        t0 = time.time()
        try:
            if arima_model is None:
                logger.warning(f"    [{symbol}] ARIMA başarısız, SARIMAX atlanıyor")
                results["sarimax"] = {"error": "ARIMA failed"}
            else:
                exog_all   = df_symbol[exog_cols].values[1:]
                exog_train = exog_all[:split_idx - 1]
                exog_test  = exog_all[split_idx - 1:]

                sarimax = SARIMAX(
                    train_returns, exog=exog_train,
                    order=arima_model.order,
                    seasonal_order=(1, 0, 1, 5),
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                sarimax_fit = sarimax.fit(disp=False, maxiter=200)
                sarimax_fc  = sarimax_fit.forecast(steps=len(test_returns), exog=exog_test)
                sarimax_price_fc = _log_returns_to_price(sarimax_fc, last_train_price)

                m = evaluate_predictions(test_returns, sarimax_fc)
                m["price_rmse"]     = round(float(np.sqrt(np.mean((test_close - sarimax_price_fc) ** 2))), 4)
                m["train_time_s"]   = round(time.time() - t0, 1)
                m["exog_variables"] = exog_cols
                m["trained_on"]     = "log_returns"

                path = registry.save_model(sarimax_fit, symbol, "sarimax", version, format="pickle")
                results["sarimax"] = {"model": sarimax_fit, "metrics": m, "path": path}
                logger.info(f"    SARIMAX: sMAPE={m.get('smape', float('nan')):.2f}% ({m['train_time_s']}s)")
        except Exception as e:
            logger.error(f"    SARIMAX failed for {symbol}: {e}")
            results["sarimax"] = {"error": str(e)}

    return results


def main():
    logger.info("=" * 60)
    logger.info("FinAI — ARIMA / SARIMA / SARIMAX Training")
    logger.info("=" * 60)

    symbols = CORE_SYMBOLS + CRYPTO_SYMBOLS
    raw_path = os.path.join(DATA_DIR, f"ohlcv_1d_{len(symbols)}sym.parquet")

    try:
        df = pd.read_parquet(raw_path)
    except FileNotFoundError:
        df = fetch_multi_symbol(symbols, period="5y", interval="1d")
        save_data(df, os.path.basename(raw_path))

    exog_cols = ["volume_ratio", "rsi_14_norm"]
    all_results = {}

    for sym in df["symbol"].unique():
        logger.info(f"\n{'='*40}\nTraining {sym}...")
        sym_data = df[df["symbol"] == sym].copy().reset_index(drop=True)

        if len(sym_data) < 200:
            logger.warning(f"Skipping {sym}: not enough data")
            continue

        sym_data["volume_ratio"] = sym_data["volume"] / (sym_data["volume"].rolling(20).mean() + 1)
        try:
            from ta.momentum import RSIIndicator
            sym_data["rsi_14_norm"] = RSIIndicator(close=sym_data["close"], window=14).rsi() / 100.0
        except Exception:
            sym_data["rsi_14_norm"] = 0.5

        sym_data = sym_data.ffill().bfill()
        results = train_arima_for_symbol(sym_data, sym, exog_cols=exog_cols)
        all_results[sym] = results

    logger.info("\n" + "=" * 60)
    logger.info("ARIMA/SARIMA/SARIMAX SUMMARY")
    logger.info("=" * 60)
    for sym, results in all_results.items():
        logger.info(f"\n  {sym}:")
        for model_name, info in results.items():
            if "error" in info:
                logger.info(f"    {model_name:8s}: FAILED — {info['error'][:60]}")
            else:
                m = info["metrics"]
                logger.info(
                    f"    {model_name:8s}: sMAPE={m.get('smape', float('nan')):.2f}% "
                    f"| RMSE={m['rmse']:.6f} | price_RMSE={m.get('price_rmse', 'N/A')} "
                    f"| IC={m.get('ic', 'N/A')}"
                )


if __name__ == "__main__":
    main()
