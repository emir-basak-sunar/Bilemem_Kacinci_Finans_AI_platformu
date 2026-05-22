"""
FinAI DL Training Utilities
============================
Paylaşılan training loop, loss, ONNX export, metrics.
train_lstm.py ve train_tcn_tft.py bu dosyadan import eder.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loss ──────────────────────────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).mean())
        return sum(losses) / len(losses)


# ── Data Preparation ──────────────────────────────────────────────────────────

def prepare_dl_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_1",
    seq_length: int = 120,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, np.ndarray]:
    """
    Prepare DataLoaders for DL training.
    Uses create_sequences_by_symbol to prevent cross-symbol leakage.
    """
    from training.data_collection import split_by_symbol
    from training.features.feature_engineering import create_sequences, create_sequences_by_symbol

    splits = split_by_symbol(df, test_ratio=0.2, val_ratio=0.1)

    scaler = StandardScaler()
    scaler.fit(splits["train"][feature_cols])

    loaders = {}
    raw_targets = {}

    for name, split in splits.items():
        split_scaled = split.copy()
        split_scaled[feature_cols] = scaler.transform(split[feature_cols])

        X_seq, y_seq = create_sequences_by_symbol(
            df=split_scaled,
            feature_cols=feature_cols,
            target_col=target_col,
            seq_length=seq_length,
            symbol_col="symbol" if "symbol" in split_scaled.columns else None,
        )

        if len(X_seq) == 0:
            X_scaled = scaler.transform(split[feature_cols])
            y = split[target_col].values
            X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

        dataset = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq))
        shuffle = (name == "train")
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        raw_targets[name] = y_seq

    return loaders["train"], loaders["val"], loaders["test"], scaler, raw_targets["test"]


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.0005,
    patience: int = 20,
    model_name: str = "model",
    is_quantile: bool = False,
    grad_accum_steps: int = 1,
    use_huber: bool = True,
) -> Dict:
    """
    Generic training loop with:
    - CosineAnnealingWarmRestarts scheduler
    - HuberLoss (outlier-robust) or QuantileLoss
    - Gradient accumulation
    - Warmup (3 epochs)
    - Early stopping
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )

    if is_quantile:
        criterion = QuantileLoss(model.quantiles)
    elif use_huber:
        criterion = nn.HuberLoss(delta=0.01)
    else:
        criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "lr": []}
    epoch         = 0

    for epoch in range(epochs):
        # Warmup
        if epoch < 3:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (epoch + 1) / 3

        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, y_batch) / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

        # Flush remaining gradients
        if len(train_loader) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)

        if epoch >= 3:
            scheduler.step(epoch - 3)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if epoch % 5 == 0:
            logger.info(f"  [{model_name}] Epoch {epoch}/{epochs} — "
                        f"train: {train_loss:.6f}, val: {val_loss:.6f}, lr: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  [{model_name}] Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1, "history": history}


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_to_onnx(model: nn.Module, input_shape: tuple, save_path: str, model_name: str = "model") -> bool:
    """
    Export PyTorch model to ONNX.
    Requires: pip install onnxscript (torch>=2.2 ile birlikte gelir)
    """
    model.eval()
    model = model.cpu()
    dummy_input = torch.randn(1, *input_shape)

    try:
        torch.onnx.export(
            model, dummy_input, save_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )
        logger.info(f"  [{model_name}] ONNX exported: {save_path}")
        return True
    except Exception as e:
        logger.warning(f"  [{model_name}] ONNX export failed: {e}")
        return False


# ── Bootstrap IC ─────────────────────────────────────────────────────────────

def bootstrap_ic(y_true: np.ndarray, y_pred: np.ndarray,
                 n_bootstrap: int = 200, ci: float = 0.90) -> Dict:
    """Bootstrap IC (Spearman) confidence interval."""
    try:
        from scipy.stats import spearmanr
        n = len(y_true)
        ics = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            ic, _ = spearmanr(y_true[idx], y_pred[idx])
            if not np.isnan(ic):
                ics.append(float(ic))
        if not ics:
            return {}
        alpha = (1 - ci) / 2
        return {
            "ic_mean": round(float(np.mean(ics)), 4),
            "ic_std":  round(float(np.std(ics)), 4),
            "ic_p5":   round(float(np.percentile(ics, alpha * 100)), 4),
            "ic_p95":  round(float(np.percentile(ics, (1 - alpha) * 100)), 4),
        }
    except Exception:
        return {}


def log_metrics(name: str, m: dict) -> None:
    """Pretty-print model metrics."""
    smape = m.get("smape", float("nan"))
    rmse  = m.get("rmse",  float("nan"))
    ic    = m.get("ic",    float("nan"))
    da    = m.get("directional_accuracy", float("nan"))
    r2    = m.get("r2",    float("nan"))

    smape_s = f"{smape:.2f}%" if isinstance(smape, float) and not np.isnan(smape) else str(smape)
    rmse_s  = f"{rmse:.6f}"  if isinstance(rmse,  float) and not np.isnan(rmse)  else str(rmse)
    ic_s    = f"{ic:.4f}"    if isinstance(ic,    float) and not np.isnan(ic)    else str(ic)
    da_s    = f"{da:.4f}"    if isinstance(da,    float) and not np.isnan(da)    else str(da)
    r2_s    = f"{r2:.4f}"    if isinstance(r2,    float) and not np.isnan(r2)    else str(r2)

    bci = m.get("bootstrap_ci", {})
    ci_str = ""
    if bci:
        lo_s = f"{bci.get('ic_p5', '?'):.3f}" if isinstance(bci.get("ic_p5"), float) else "?"
        hi_s = f"{bci.get('ic_p95', '?'):.3f}" if isinstance(bci.get("ic_p95"), float) else "?"
        ci_str = f" | IC_CI: [{lo_s}, {hi_s}]"

    ci_width = m.get("confidence_interval_width")
    ci_width_str = f" | CI_width: {ci_width:.4f}" if ci_width is not None else ""

    logger.info(
        f"  {name:6s} | sMAPE: {smape_s} | RMSE: {rmse_s} "
        f"| Dir.Acc: {da_s} | IC: {ic_s} | R²: {r2_s}"
        f"{ci_str}{ci_width_str}"
    )
