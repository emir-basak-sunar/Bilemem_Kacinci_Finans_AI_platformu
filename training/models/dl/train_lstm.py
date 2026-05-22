"""
FinAI DL — BiLSTM Training
============================
Bidirectional LSTM with LayerNorm + GELU.
A100 için: hidden=256, layers=3, batch=256, seq=120.

Çalıştırma:
    python -m training.models.dl.train_lstm
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
    FEATURE_CFG, LSTM_CFG, DATA_DIR, SYMBOL_ENCODING, CRYPTO_SYMBOLS,
)
from training.features.feature_engineering import get_feature_columns
from training.model_registry import ModelRegistry, ModelManifest
from training.models.tree.train_tree_models import evaluate_predictions
from training.walk_forward import compute_quant_metrics
from training.models.dl.architectures import BiLSTM
from training.models.dl.train_utils import (
    DEVICE, prepare_dl_data, train_model, export_to_onnx,
    bootstrap_ic, log_metrics,
)

import torch
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_lstm(df: pd.DataFrame, feature_cols: list, symbol: str = "EQUITY_UNIVERSAL") -> dict:
    """Train BiLSTM and return metrics + saved paths."""
    seq_length = FEATURE_CFG.sequence_length
    registry   = ModelRegistry()
    version    = registry.get_next_version(symbol)

    logger.info(f"\nPreparing sequences (seq_length={seq_length})...")
    train_loader, val_loader, test_loader, scaler, y_test = prepare_dl_data(
        df, feature_cols, "target_1", seq_length, batch_size=LSTM_CFG.batch_size,
    )

    input_size = len(feature_cols)
    lstm = BiLSTM(
        input_size=input_size,
        hidden_size=LSTM_CFG.hidden_size,
        num_layers=LSTM_CFG.num_layers,
        dropout=LSTM_CFG.dropout,
        bidirectional=LSTM_CFG.bidirectional,
    )

    logger.info(f"Training BiLSTM (hidden={LSTM_CFG.hidden_size}, layers={LSTM_CFG.num_layers})...")
    train_model(
        lstm, train_loader, val_loader,
        epochs=LSTM_CFG.epochs, lr=LSTM_CFG.lr,
        patience=LSTM_CFG.patience, model_name="LSTM", use_huber=True,
    )

    lstm.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.extend(lstm(X_batch.to(DEVICE)).cpu().numpy())
    preds = np.array(preds)

    metrics = evaluate_predictions(y_test, preds)
    metrics["bootstrap_ci"] = bootstrap_ic(y_test, preds)
    log_metrics("LSTM", metrics)
    try:
        metrics["quant_metrics"] = compute_quant_metrics(y_test, preds)
    except Exception:
        pass

    lstm_config = {
        "input_size": input_size,
        "hidden_size": LSTM_CFG.hidden_size,
        "num_layers": LSTM_CFG.num_layers,
        "dropout": LSTM_CFG.dropout,
        "bidirectional": LSTM_CFG.bidirectional,
    }
    registry.save_model(lstm.state_dict(), symbol, "lstm", version, format="pt", model_config=lstm_config)

    onnx_path = os.path.join(registry.models_dir, "lstm", f"{symbol}_v{version}.onnx")
    export_to_onnx(lstm, (seq_length, input_size), onnx_path, "LSTM")

    # Save scaler
    scaler_filename = f"scaler_{symbol}_v{version}.joblib"
    scaler_path = os.path.join(registry.models_dir, scaler_filename)
    joblib.dump(scaler, scaler_path)

    return {
        "metrics": metrics,
        "version": version,
        "scaler_path": scaler_path,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "preds": preds,
    }


def main():
    logger.info("=" * 60)
    logger.info(f"FinAI — BiLSTM Training | Device: {DEVICE}")
    logger.info("=" * 60)

    featured_path = os.path.join(DATA_DIR, "featured_equity_1d.parquet")
    if not os.path.exists(featured_path):
        featured_path = os.path.join(DATA_DIR, "featured_universal_1d.parquet")
        if not os.path.exists(featured_path):
            logger.error("Featured data not found! Run train_tree_models first.")
            return

    df = pd.read_parquet(featured_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    if "symbol" in df.columns:
        crypto_mask = df["symbol"].isin(CRYPTO_SYMBOLS)
        if crypto_mask.sum() > 0:
            df = df[~crypto_mask].copy()

    feature_cols = get_feature_columns(df, sma_windows=FEATURE_CFG.sma_windows, ema_windows=FEATURE_CFG.ema_windows, include_macro=True)
    if "symbol" in df.columns:
        df = df.copy()
        df["symbol_encoded"] = df["symbol"].map(SYMBOL_ENCODING).fillna(-1).astype(int)
        if "symbol_encoded" not in feature_cols:
            feature_cols.append("symbol_encoded")

    logger.info(f"Feature count: {len(feature_cols)}")
    result = train_lstm(df, feature_cols)

    logger.info("\n" + "=" * 60)
    logger.info("LSTM TRAINING COMPLETE")
    log_metrics("LSTM", result["metrics"])
    qm = result["metrics"].get("quant_metrics")
    if qm:
        logger.info(f"  Sharpe(L): {qm['sharpe']:.3f} | Sharpe(LS): {qm['ls_sharpe']:.3f} | "
                    f"MaxDD: {qm['max_drawdown']:.3f}")


if __name__ == "__main__":
    main()
