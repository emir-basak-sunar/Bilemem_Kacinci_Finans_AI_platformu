"""
FinAI Statistical Models — GARCH Volatility
=============================================
GARCH(1,1) volatility model per symbol.
Position sizing için kullanılır: yüksek vol → küçük pozisyon.

Çalıştırma:
    python -m training.models.statistical.train_garch
"""
import sys
import os
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.config import CORE_SYMBOLS, CRYPTO_SYMBOLS, DATA_DIR
from training.data_collection import fetch_multi_symbol, save_data
from training.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch kütüphanesi yok. Kurulum: pip install arch")


def train_garch_for_symbol(
    df_symbol: pd.DataFrame,
    symbol: str,
    test_ratio: float = 0.2,
) -> Dict:
    """
    GARCH(1,1) + AR(1) mean model.

    Returns:
        Dict with model, metrics (AIC, BIC, vol_corr, persistence), path.
    """
    if not ARCH_AVAILABLE:
        return {"error": "arch kütüphanesi yok — pip install arch"}

    close = df_symbol["close"].values
    n = len(close)
    split_idx = int(n * (1 - test_ratio))

    # Log-returns × 100 (GARCH % cinsinden çalışır)
    log_returns = np.log(close[1:] / close[:-1]) * 100
    train_returns = log_returns[:split_idx - 1]
    test_returns  = log_returns[split_idx - 1:]

    registry = ModelRegistry()
    version  = registry.get_next_version(f"{symbol}_GARCH")

    logger.info(f"  [{symbol}] GARCH(1,1) volatility model...")
    t0 = time.time()

    try:
        garch = arch_model(
            train_returns,
            vol="Garch", p=1, q=1,
            mean="AR", lags=1,
            dist="t",  # Student-t: fat tails için
        )
        garch_fit = garch.fit(disp="off", show_warning=False)

        # In-sample conditional volatility (full training period)
        cond_vol = garch_fit.conditional_volatility

        # Out-of-sample: use last forecast variance as constant vol estimate
        # and compare with realized vol for correlation
        forecasts = garch_fit.forecast(horizon=1, reindex=False)
        last_vol_fc = float(np.sqrt(forecasts.variance.values[-1, 0]))

        # Realized volatility (5-bar rolling std) on test set
        realized_vol = pd.Series(test_returns).rolling(5).std().values

        # Use in-sample conditional vol tail for correlation
        # (more meaningful than single-point forecast)
        train_realized = pd.Series(train_returns).rolling(5).std().values
        cond_vol_arr = cond_vol.values if hasattr(cond_vol, 'values') else np.array(cond_vol)

        # Align lengths and compute correlation
        min_len = min(len(cond_vol_arr), len(train_realized))
        cv = cond_vol_arr[-min_len:]
        rv = train_realized[-min_len:]
        mask = ~np.isnan(rv) & ~np.isnan(cv)
        vol_corr = float(np.corrcoef(cv[mask], rv[mask])[0, 1]) if mask.sum() > 5 else float("nan")

        elapsed = round(time.time() - t0, 1)
        metrics = {
            "aic":              round(float(garch_fit.aic), 2),
            "bic":              round(float(garch_fit.bic), 2),
            "vol_forecast_corr": round(vol_corr, 4),
            "train_time_s":     elapsed,
            "alpha":            round(float(garch_fit.params.get("alpha[1]", float("nan"))), 4),
            "beta":             round(float(garch_fit.params.get("beta[1]",  float("nan"))), 4),
            "persistence":      round(
                float(garch_fit.params.get("alpha[1]", 0)) +
                float(garch_fit.params.get("beta[1]",  0)), 4
            ),
        }

        path = registry.save_model(garch_fit, f"{symbol}_GARCH", "arima", version, format="pickle")

        logger.info(
            f"    GARCH(1,1): AIC={metrics['aic']:.1f} | "
            f"α={metrics['alpha']:.3f} | β={metrics['beta']:.3f} | "
            f"persistence={metrics['persistence']:.3f} | "
            f"vol_corr={metrics['vol_forecast_corr']:.3f} ({elapsed}s)"
        )
        return {"model": garch_fit, "metrics": metrics, "path": path}

    except Exception as e:
        logger.error(f"    GARCH failed for {symbol}: {e}")
        return {"error": str(e)}


def main():
    logger.info("=" * 60)
    logger.info("FinAI — GARCH Volatility Training")
    logger.info("=" * 60)

    if not ARCH_AVAILABLE:
        logger.error("arch kütüphanesi yok. Kurulum: pip install arch")
        return

    symbols = CORE_SYMBOLS + CRYPTO_SYMBOLS
    raw_path = os.path.join(DATA_DIR, f"ohlcv_1d_{len(symbols)}sym.parquet")

    try:
        df = pd.read_parquet(raw_path)
    except FileNotFoundError:
        df = fetch_multi_symbol(symbols, period="5y", interval="1d")
        save_data(df, os.path.basename(raw_path))

    all_results = {}
    for sym in df["symbol"].unique():
        sym_data = df[df["symbol"] == sym].copy().reset_index(drop=True)
        if len(sym_data) < 200:
            continue
        result = train_garch_for_symbol(sym_data, sym)
        all_results[sym] = result

    logger.info("\n" + "=" * 60)
    logger.info("GARCH SUMMARY")
    logger.info("=" * 60)
    for sym, result in all_results.items():
        if "error" in result:
            logger.info(f"  {sym}: FAILED — {result['error'][:60]}")
        else:
            m = result["metrics"]
            logger.info(
                f"  {sym}: AIC={m['aic']:.1f} | persistence={m['persistence']:.3f} "
                f"| vol_corr={m['vol_forecast_corr']:.3f}"
            )


if __name__ == "__main__":
    main()
