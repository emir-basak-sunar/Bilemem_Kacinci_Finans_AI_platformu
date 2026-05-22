"""
FinAI DL — TCN + TFT Training
================================
TCN: 8 katman, receptive field ~1020 bar.
TFT: hidden=128, 8 head, 7 quantile (P5-P95).

Çalıştırma:
    python -m training.models.dl.train_tcn_tft
"""
import sys
import os
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.config import (
    FEATURE_CFG, TCN_CFG, TFT_CFG, DATA_DIR, SYMBOL_ENCODING, CRYPTO_SYMBOLS,
)
from training.features.feature_engineering import get_feature_columns
from training.model_registry import ModelRegistry
from training.models.tree.train_tree_models import evaluate_predictions
from training.walk_forward import compute_quant_metrics
from training.models.dl.architectures import TCN, TFTModel
from training.models.dl.train_utils import (
    DEVICE, prepare_dl_data, train_model, export_to_onnx,
    bootstrap_ic, log_metrics,
)

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_tcn(df: pd.DataFrame, feature_cols: list, symbol: str = "EQUITY_UNIVERSAL") -> dict:
    """Train TCN and return metrics."""
    seq_length = FEATURE_CFG.sequence_length
    registry   = ModelRegistry()
    version    = registry.get_next_version(symbol)

    train_loader, val_loader, test_loader, scaler, y_test = prepare_dl_data(
        df, feature_cols, "target_1", seq_length, batch_size=TCN_CFG.batch_size,
    )

    input_size = len(feature_cols)
    tcn = TCN(
        input_size=input_size,
        num_channels=TCN_CFG.num_channels,
        kernel_size=TCN_CFG.kernel_size,
        dropout=TCN_CFG.dropout,
    )

    logger.info(f"Training TCN ({len(TCN_CFG.num_channels)} layers, receptive field ~1020 bars)...")
    train_model(
        tcn, train_loader, val_loader,
        epochs=TCN_CFG.epochs, lr=TCN_CFG.lr,
        patience=TCN_CFG.patience, model_name="TCN", use_huber=True,
    )

    tcn.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.extend(tcn(X_batch.to(DEVICE)).cpu().numpy())
    preds = np.array(preds)

    metrics = evaluate_predictions(y_test, preds)
    metrics["bootstrap_ci"] = bootstrap_ic(y_test, preds)
    log_metrics("TCN", metrics)
    try:
        metrics["quant_metrics"] = compute_quant_metrics(y_test, preds)
    except Exception:
        pass

    tcn_config = {
        "input_size": input_size,
        "num_channels": TCN_CFG.num_channels,
        "kernel_size": TCN_CFG.kernel_size,
        "dropout": TCN_CFG.dropout,
    }
    registry.save_model(tcn.state_dict(), symbol, "tcn", version, format="pt", model_config=tcn_config)
    onnx_path = os.path.join(registry.models_dir, "tcn", f"{symbol}_v{version}.onnx")
    export_to_onnx(tcn, (seq_length, input_size), onnx_path, "TCN")

    return {"metrics": metrics, "version": version, "y_test": y_test, "preds": preds}


def train_tft(df: pd.DataFrame, feature_cols: list, symbol: str = "EQUITY_UNIVERSAL") -> dict:
    """Train TFT with quantile outputs."""
    seq_length = FEATURE_CFG.sequence_length
    registry   = ModelRegistry()
    version    = registry.get_next_version(symbol)

    train_loader, val_loader, test_loader, scaler, y_test = prepare_dl_data(
        df, feature_cols, "target_1", seq_length, batch_size=TFT_CFG.batch_size,
    )

    input_size = len(feature_cols)
    tft = TFTModel(
        input_size=input_size,
        hidden_size=TFT_CFG.hidden_size,
        n_heads=TFT_CFG.attention_head_size,
        dropout=TFT_CFG.dropout,
        quantiles=TFT_CFG.quantiles,
    )

    logger.info(f"Training TFT (hidden={TFT_CFG.hidden_size}, heads={TFT_CFG.attention_head_size}, "
                f"{len(TFT_CFG.quantiles)} quantiles)...")
    train_model(
        tft, train_loader, val_loader,
        epochs=TFT_CFG.epochs, lr=TFT_CFG.lr,
        patience=TFT_CFG.patience, model_name="TFT", is_quantile=True,
    )

    tft.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            all_preds.append(tft(X_batch.to(DEVICE)).cpu().numpy())

    preds_full = np.concatenate(all_preds, axis=0)
    median_idx = TFT_CFG.quantiles.index(0.5)
    tft_median = preds_full[:, median_idx]

    metrics = evaluate_predictions(y_test, tft_median)
    metrics["bootstrap_ci"] = bootstrap_ic(y_test, tft_median)

    p10_idx = TFT_CFG.quantiles.index(0.1)
    p90_idx = TFT_CFG.quantiles.index(0.9)
    metrics["confidence_interval_width"] = round(
        float(np.mean(preds_full[:, p90_idx] - preds_full[:, p10_idx])), 6
    )
    log_metrics("TFT", metrics)
    try:
        metrics["quant_metrics"] = compute_quant_metrics(y_test, tft_median)
    except Exception:
        pass

    tft_config = {
        "input_size": input_size,
        "hidden_size": TFT_CFG.hidden_size,
        "n_heads": TFT_CFG.attention_head_size,
        "dropout": TFT_CFG.dropout,
        "quantiles": TFT_CFG.quantiles,
    }
    registry.save_model(tft.state_dict(), symbol, "tft", version, format="pt", model_config=tft_config)
    onnx_path = os.path.join(registry.models_dir, "tft", f"{symbol}_v{version}.onnx")
    export_to_onnx(tft, (seq_length, input_size), onnx_path, "TFT")

    return {
        "metrics": metrics, "version": version,
        "y_test": y_test, "preds": tft_median, "preds_full": preds_full,
    }


def main():
    logger.info("=" * 60)
    logger.info(f"FinAI — TCN + TFT Training | Device: {DEVICE}")
    logger.info("=" * 60)

    featured_path = os.path.join(DATA_DIR, "featured_equity_1d.parquet")
    if not os.path.exists(featured_path):
        featured_path = os.path.join(DATA_DIR, "featured_universal_1d.parquet")
        if not os.path.exists(featured_path):
            logger.error("Featured data not found! Run train_tree_models first.")
            return

    df = pd.read_parquet(featured_path)
    if "symbol" in df.columns:
        df = df[~df["symbol"].isin(CRYPTO_SYMBOLS)].copy()

    feature_cols = get_feature_columns(df, sma_windows=FEATURE_CFG.sma_windows, ema_windows=FEATURE_CFG.ema_windows, include_macro=True)
    if "symbol" in df.columns:
        df = df.copy()
        df["symbol_encoded"] = df["symbol"].map(SYMBOL_ENCODING).fillna(-1).astype(int)
        if "symbol_encoded" not in feature_cols:
            feature_cols.append("symbol_encoded")

    logger.info(f"Feature count: {len(feature_cols)}")

    # TCN
    logger.info("\n" + "=" * 40)
    tcn_result = train_tcn(df, feature_cols)

    # TFT
    logger.info("\n" + "=" * 40)
    tft_result = train_tft(df, feature_cols)

    logger.info("\n" + "=" * 60)
    logger.info("TCN + TFT TRAINING COMPLETE")
    log_metrics("TCN", tcn_result["metrics"])
    log_metrics("TFT", tft_result["metrics"])


if __name__ == "__main__":
    main()
