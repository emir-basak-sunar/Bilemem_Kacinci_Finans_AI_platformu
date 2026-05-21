"""
FinAI MLflow Service
=====================
trained_models/ klasöründeki tüm manifest + metrics dosyalarını
MLflow'a import eder ve canlı model durumunu sorgular.

MLflow UI: http://localhost:5000
Tracking URI: sqlite:///mlflow.db (local) veya MLFLOW_TRACKING_URI env var

Kullanım:
    # İlk kez import (tüm modelleri MLflow'a yükle)
    python -m backend.services.mlflow_service

    # Sadece servis olarak kullan (FastAPI endpoint'lerinden)
    from services.mlflow_service import get_mlflow_summary, get_experiment_runs
"""
import os
import sys
import json
import glob
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ── MLflow tracking URI ────────────────────────────────────────────────────────
# Workspace root'ta mlflow.db oluşturulur
_BACKEND_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_DIR = os.path.dirname(_BACKEND_DIR)
MLFLOW_DB_PATH = os.path.join(_WORKSPACE_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH}"
)

# trained_models/ registry
TRAINED_MODELS_DIR = os.path.join(_WORKSPACE_DIR, "trained_models")
MANIFESTS_DIR = os.path.join(TRAINED_MODELS_DIR, "manifests")
METRICS_DIR   = os.path.join(TRAINED_MODELS_DIR, "metrics")

# MLflow experiment adı
EXPERIMENT_NAME = "FinAI-Production-Models"


def _get_mlflow():
    """MLflow client'ı lazy import ile döner."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return mlflow
    except ImportError:
        logger.warning("mlflow not installed. Run: pip install mlflow")
        return None


def import_all_models_to_mlflow() -> Dict[str, Any]:
    """
    trained_models/ klasöründeki tüm manifest + metrics dosyalarını
    MLflow'a run olarak import eder.

    Her manifest → 1 MLflow run
    Her model (xgboost/lightgbm/catboost/lstm/tcn/tft) → ayrı tag + metric set

    Returns:
        {"imported": int, "skipped": int, "errors": list}
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return {"error": "mlflow not installed"}

    # Experiment oluştur veya al
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "FinAI", "type": "production"}
        )
    else:
        experiment_id = experiment.experiment_id

    manifest_files = sorted(glob.glob(os.path.join(MANIFESTS_DIR, "*_v*.json")))
    imported = 0
    skipped  = 0
    errors   = []

    for manifest_path in manifest_files:
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            symbol  = manifest["symbol"]
            version = manifest["version"]
            run_name = f"{symbol}_v{version}"

            # Zaten import edilmiş mi kontrol et
            existing = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.run_name = '{run_name}'",
                max_results=1,
            )
            if len(existing) > 0:
                logger.debug(f"Skipping already imported: {run_name}")
                skipped += 1
                continue

            # Backtest metrics dosyasını yükle
            backtest_path = os.path.join(METRICS_DIR, f"{symbol}_v{version}_backtest.json")
            backtest_data = {}
            if os.path.exists(backtest_path):
                with open(backtest_path) as f:
                    backtest_data = json.load(f)

            # SHAP dosyasını yükle
            shap_path = os.path.join(METRICS_DIR, f"{symbol}_v{version}_shap.json")
            shap_data = []
            if os.path.exists(shap_path):
                with open(shap_path) as f:
                    shap_data = json.load(f)

            # MLflow run başlat
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
            ) as run:
                # ── Tags ──
                mlflow.set_tags({
                    "run_name":    run_name,
                    "symbol":      symbol,
                    "version":     str(version),
                    "trained_at":  manifest.get("trained_at", "unknown"),
                    "data_range_start": manifest.get("data_range", ["?", "?"])[0],
                    "data_range_end":   manifest.get("data_range", ["?", "?"])[1],
                    "horizons":    str(manifest.get("horizons", [])),
                    "n_features":  str(len(manifest.get("feature_columns", []))),
                    "model_types": str(list(manifest.get("models", {}).keys())),
                    "project":     "FinAI",
                    "stage":       "production",
                })

                # ── Ensemble weights ──
                ew = manifest.get("ensemble_weights", {})
                for model_name, weight in ew.items():
                    mlflow.log_metric(f"ensemble_weight_{model_name}", weight)

                # ── Per-model metrics (manifest'ten) ──
                for model_name, model_info in manifest.get("models", {}).items():
                    if isinstance(model_info, dict):
                        for metric_name in ["mae", "rmse", "smape", "directional_accuracy", "ic", "r2", "mape"]:
                            val = model_info.get(metric_name)
                            if val is not None and isinstance(val, (int, float)):
                                mlflow.log_metric(f"{model_name}_{metric_name}", float(val))
                        n_samples = model_info.get("n_samples")
                        if n_samples:
                            mlflow.log_metric(f"{model_name}_n_samples", float(n_samples))
                        train_time = model_info.get("train_time_s")
                        if train_time:
                            mlflow.log_metric(f"{model_name}_train_time_s", float(train_time))

                # ── Backtest metrics (ensemble + quant) ──
                for section_name, section_data in backtest_data.items():
                    if isinstance(section_data, dict):
                        for metric_name, val in section_data.items():
                            if isinstance(val, (int, float)):
                                mlflow.log_metric(f"backtest_{section_name}_{metric_name}", float(val))

                # ── Ensemble backtest summary (top-level) ──
                bm = manifest.get("backtest_metrics", {})
                for metric_name, val in bm.items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(f"ensemble_{metric_name}", float(val))

                # ── SHAP top features ──
                if shap_data:
                    for i, item in enumerate(shap_data[:10]):
                        feat  = item.get("feature", f"feat_{i}")
                        shap_val = item.get("shap_mean_abs", 0)
                        mlflow.log_metric(f"shap_rank_{i+1:02d}_{feat[:30]}", float(shap_val))

                # ── Manifest JSON artifact ──
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump(manifest, tmp, indent=2, default=str)
                    tmp_path = tmp.name
                mlflow.log_artifact(tmp_path, artifact_path="manifest")
                os.unlink(tmp_path)

                # ── SHAP artifact ──
                if shap_data:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                        json.dump(shap_data, tmp, indent=2)
                        tmp_path = tmp.name
                    mlflow.log_artifact(tmp_path, artifact_path="shap")
                    os.unlink(tmp_path)

            logger.info(f"Imported to MLflow: {run_name}")
            imported += 1

        except Exception as e:
            logger.error(f"Failed to import {manifest_path}: {e}")
            errors.append({"file": os.path.basename(manifest_path), "error": str(e)})

    return {
        "imported": imported,
        "skipped":  skipped,
        "errors":   errors,
        "experiment_id": experiment_id,
        "tracking_uri":  MLFLOW_TRACKING_URI,
    }


def get_mlflow_summary() -> Dict[str, Any]:
    """
    MLflow'daki tüm production run'ların özetini döner.
    FastAPI /api/mlflow/summary endpoint'i tarafından kullanılır.
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return {"error": "mlflow not installed", "runs": []}

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return {"runs": [], "message": "No experiments found. Call /api/mlflow/import first."}

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["tags.symbol ASC", "tags.version DESC"],
            max_results=200,
        )

        if runs_df.empty:
            return {"runs": [], "experiment_id": experiment.experiment_id}

        runs = []
        for _, row in runs_df.iterrows():
            run_dict = {
                "run_id":      row.get("run_id", ""),
                "run_name":    row.get("tags.run_name", ""),
                "symbol":      row.get("tags.symbol", ""),
                "version":     row.get("tags.version", ""),
                "trained_at":  row.get("tags.trained_at", ""),
                "stage":       row.get("tags.stage", "production"),
                "n_features":  _safe_int(row.get("tags.n_features")),
                "model_types": row.get("tags.model_types", ""),
                "horizons":    row.get("tags.horizons", ""),
                "metrics": {
                    # Ensemble metrics
                    "ic":                   _safe_float(row.get("metrics.ensemble_ic")),
                    "directional_accuracy": _safe_float(row.get("metrics.ensemble_directional_accuracy")),
                    "mae":                  _safe_float(row.get("metrics.ensemble_mae")),
                    "rmse":                 _safe_float(row.get("metrics.ensemble_rmse")),
                    "smape":                _safe_float(row.get("metrics.ensemble_smape")),
                    "r2":                   _safe_float(row.get("metrics.ensemble_r2")),
                    # Quant metrics
                    "sharpe":               _safe_float(row.get("metrics.backtest_ensemble_quant_sharpe")),
                    "sortino":              _safe_float(row.get("metrics.backtest_ensemble_quant_sortino")),
                    "calmar":               _safe_float(row.get("metrics.backtest_ensemble_quant_calmar")),
                    "max_drawdown":         _safe_float(row.get("metrics.backtest_ensemble_quant_max_drawdown")),
                    "annual_return":        _safe_float(row.get("metrics.backtest_ensemble_quant_annual_return")),
                    "ls_sharpe":            _safe_float(row.get("metrics.backtest_ensemble_quant_ls_sharpe")),
                    "ls_sortino":           _safe_float(row.get("metrics.backtest_ensemble_quant_ls_sortino")),
                    # Per-model IC
                    "xgboost_ic":           _safe_float(row.get("metrics.xgboost_ic")),
                    "lightgbm_ic":          _safe_float(row.get("metrics.lightgbm_ic")),
                    "catboost_ic":          _safe_float(row.get("metrics.catboost_ic")),
                    "lstm_directional_accuracy": _safe_float(row.get("metrics.lstm_directional_accuracy")),
                    "tcn_directional_accuracy":  _safe_float(row.get("metrics.tcn_directional_accuracy")),
                    "tft_directional_accuracy":  _safe_float(row.get("metrics.tft_directional_accuracy")),
                },
                "ensemble_weights": {
                    "xgboost":  _safe_float(row.get("metrics.ensemble_weight_xgboost")),
                    "lightgbm": _safe_float(row.get("metrics.ensemble_weight_lightgbm")),
                    "catboost": _safe_float(row.get("metrics.ensemble_weight_catboost")),
                },
            }
            runs.append(run_dict)

        return {
            "runs":          runs,
            "total":         len(runs),
            "experiment_id": experiment.experiment_id,
            "tracking_uri":  MLFLOW_TRACKING_URI,
            "mlflow_ui_url": "http://localhost:5000",
        }

    except Exception as e:
        logger.error(f"MLflow summary failed: {e}")
        return {"error": str(e), "runs": []}


def get_run_detail(run_name: str) -> Dict[str, Any]:
    """Belirli bir run'ın tüm metriklerini ve SHAP verilerini döner."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return {"error": "mlflow not installed"}

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return {"error": "No experiment found"}

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.run_name = '{run_name}'",
            max_results=1,
        )

        if runs_df.empty:
            return {"error": f"Run '{run_name}' not found"}

        row = runs_df.iloc[0]
        run_id = row["run_id"]

        # Tüm metric'leri al
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        metrics = {k: v for k, v in run.data.metrics.items()}
        tags    = {k: v for k, v in run.data.tags.items()
                   if not k.startswith("mlflow.")}

        # SHAP artifact'ı oku
        shap_data = []
        try:
            artifacts = client.list_artifacts(run_id, path="shap")
            if artifacts:
                import tempfile
                local_path = client.download_artifacts(run_id, "shap", tempfile.gettempdir())
                shap_file = os.path.join(local_path, "tmp.json")
                # Artifact adını bul
                for art in artifacts:
                    art_local = client.download_artifacts(run_id, art.path, tempfile.gettempdir())
                    with open(art_local) as f:
                        shap_data = json.load(f)
                    break
        except Exception:
            pass

        return {
            "run_id":   run_id,
            "run_name": run_name,
            "tags":     tags,
            "metrics":  metrics,
            "shap":     shap_data,
        }

    except Exception as e:
        logger.error(f"MLflow run detail failed: {e}")
        return {"error": str(e)}


def get_model_comparison() -> Dict[str, Any]:
    """
    Tüm model ailelerini karşılaştırır.
    IC, Sharpe, Directional Accuracy bazında sıralama yapar.
    """
    summary = get_mlflow_summary()
    if "error" in summary:
        return summary

    runs = summary.get("runs", [])
    if not runs:
        return {"comparison": [], "message": "No runs found"}

    # En iyi IC'ye göre sırala
    sorted_by_ic = sorted(
        [r for r in runs if r["metrics"].get("ic") is not None],
        key=lambda r: r["metrics"]["ic"],
        reverse=True,
    )

    # En iyi Sharpe'a göre sırala
    sorted_by_sharpe = sorted(
        [r for r in runs if r["metrics"].get("sharpe") is not None],
        key=lambda r: r["metrics"]["sharpe"],
        reverse=True,
    )

    return {
        "by_ic":     [{"run_name": r["run_name"], "ic": r["metrics"]["ic"],
                       "dir_acc": r["metrics"]["directional_accuracy"]} for r in sorted_by_ic[:10]],
        "by_sharpe": [{"run_name": r["run_name"], "sharpe": r["metrics"]["sharpe"],
                       "ls_sharpe": r["metrics"]["ls_sharpe"]} for r in sorted_by_sharpe[:10]],
        "total_runs": len(runs),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            return None
        return round(float(val), 6)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ── CLI: python -m backend.services.mlflow_service ────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Trained Models Dir:  {TRAINED_MODELS_DIR}")
    logger.info("Importing all models to MLflow...")
    result = import_all_models_to_mlflow()
    logger.info(f"Import complete: {result}")
    logger.info("Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view the UI")
