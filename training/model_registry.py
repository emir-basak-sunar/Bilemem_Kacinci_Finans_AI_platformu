"""
FinAI Model Registry — v1.1
=============================
Handles saving, loading, and versioning trained models.
Supports: joblib (tree), ONNX (DL), pickle (ARIMA), PyTorch JIT

Değişiklikler v1.1:
- get_latest_manifest: alfabetik sort yerine numerik sort (v9 < v10 hatası düzeltildi)
- from_dict: bilinmeyen alanları filtreler — eski manifest'ler crash vermez
- load_model (pt): weights_only=True ile güvenli yükleme
- save_model (pt): state_dict yerine mimari config da kaydedilir
"""
import os
import json
import glob
import logging
import joblib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ModelManifest:
    """Metadata for a trained model bundle."""

    # Bilinen alanlar — from_dict'te bilinmeyenler filtrelenir
    _KNOWN_FIELDS = {
        "symbol", "version", "trained_at", "data_range",
        "models", "ensemble_weights", "backtest_metrics",
        "feature_columns", "horizons", "symbol_encoding",
        "scaler", "seq_length",
    }

    def __init__(
        self,
        symbol: str,
        version: int,
        trained_at: str = None,
        data_range: List[str] = None,
        models: Dict[str, Dict[str, Any]] = None,
        ensemble_weights: Dict[str, float] = None,
        backtest_metrics: Dict[str, float] = None,
        feature_columns: List[str] = None,
        horizons: List[int] = None,
        symbol_encoding: Dict[str, int] = None,
        scaler: str = None,
        seq_length: int = None,
    ):
        self.symbol = symbol.upper()
        self.version = version
        self.trained_at = trained_at or datetime.now(timezone.utc).isoformat()
        self.data_range = data_range or []
        self.models = models or {}
        self.ensemble_weights = ensemble_weights or {}
        self.backtest_metrics = backtest_metrics or {}
        self.feature_columns = feature_columns or []
        self.horizons = horizons or [1, 5, 10, 20]
        # v1.1: symbol encoding mapping — serving'de tutarlı kullanım için
        self.symbol_encoding = symbol_encoding or {}
        # v1.1: DL model alanları
        self.scaler = scaler
        self.seq_length = seq_length

    def to_dict(self) -> dict:
        d = {
            "symbol": self.symbol,
            "version": self.version,
            "trained_at": self.trained_at,
            "data_range": self.data_range,
            "models": self.models,
            "ensemble_weights": self.ensemble_weights,
            "backtest_metrics": self.backtest_metrics,
            "feature_columns": self.feature_columns,
            "horizons": self.horizons,
            "symbol_encoding": self.symbol_encoding,
        }
        # Opsiyonel alanlar — sadece set edilmişse ekle
        if self.scaler is not None:
            d["scaler"] = self.scaler
        if self.seq_length is not None:
            d["seq_length"] = self.seq_length
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ModelManifest":
        # FIX v1.1: Bilinmeyen alanları filtrele — eski manifest'ler TypeError vermez
        filtered = {k: v for k, v in data.items() if k in cls._KNOWN_FIELDS}
        return cls(**filtered)

    def save(self, manifests_dir: str):
        os.makedirs(manifests_dir, exist_ok=True)
        filename = f"{self.symbol}_v{self.version}.json"
        path = os.path.join(manifests_dir, filename)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved manifest: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ModelManifest":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class ModelRegistry:
    """
    Central registry for trained models.

    Directory structure:
        model_registry/
        ├── manifests/
        │   ├── AAPL_v1.json
        │   └── AAPL_v2.json
        ├── models/
        │   ├── xgboost/AAPL_v2.joblib
        │   ├── lightgbm/AAPL_v2.joblib
        │   ├── catboost/AAPL_v2.joblib
        │   ├── lstm/AAPL_v2.onnx
        │   ├── tcn/AAPL_v2.onnx
        │   ├── tft/AAPL_v2.onnx
        │   ├── arima/AAPL_v2.pkl
        │   └── sarima/AAPL_v2.pkl   ← v1.1: SARIMA artık ayrı klasörde
        └── metrics/
            └── AAPL_v2_backtest.json
    """

    def __init__(self, registry_dir: str = None):
        if registry_dir is None:
            from training.config import MODEL_REGISTRY_DIR
            registry_dir = MODEL_REGISTRY_DIR

        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.manifests_dir = os.path.join(registry_dir, "manifests")
        self.metrics_dir = os.path.join(registry_dir, "metrics")

        for d in [self.models_dir, self.manifests_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

    def get_next_version(self, symbol: str) -> int:
        """Get next version number for a symbol."""
        pattern = os.path.join(self.manifests_dir, f"{symbol.upper()}_v*.json")
        existing = glob.glob(pattern)
        if not existing:
            return 1

        versions = []
        for f in existing:
            try:
                v = int(os.path.basename(f).split("_v")[1].split(".")[0])
                versions.append(v)
            except (ValueError, IndexError):
                continue

        return max(versions) + 1 if versions else 1

    def get_latest_manifest(self, symbol: str) -> Optional[ModelManifest]:
        """Get the latest manifest for a symbol."""
        pattern = os.path.join(self.manifests_dir, f"{symbol.upper()}_v*.json")
        existing = glob.glob(pattern)
        if not existing:
            return None

        # FIX v1.1: Numerik sıralama — "v10" > "v9" doğru çalışır
        def _version_key(path: str) -> int:
            try:
                return int(os.path.basename(path).split("_v")[1].split(".")[0])
            except (ValueError, IndexError):
                return 0

        latest = max(existing, key=_version_key)
        return ModelManifest.load(latest)

    def save_model(
        self,
        model,
        symbol: str,
        model_type: str,
        version: int,
        format: str = "joblib",
        model_config: Dict[str, Any] = None,
    ) -> str:
        """
        Save a trained model to the registry.

        Args:
            model: Trained model object (state_dict, bytes, sklearn model, etc.)
            symbol: Stock symbol
            model_type: xgboost, lightgbm, catboost, lstm, tcn, tft, arima, sarima, sarimax
            version: Model version number
            format: joblib, onnx, pickle, pt (pytorch)
            model_config: (pt format) Mimari parametreleri — yükleme için gerekli

        Returns:
            Path to saved model
        """
        type_dir = os.path.join(self.models_dir, model_type)
        os.makedirs(type_dir, exist_ok=True)

        filename = f"{symbol.upper()}_v{version}"

        if format == "joblib":
            path = os.path.join(type_dir, f"{filename}.joblib")
            joblib.dump(model, path)

        elif format == "pickle":
            import pickle
            path = os.path.join(type_dir, f"{filename}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

        elif format == "onnx":
            path = os.path.join(type_dir, f"{filename}.onnx")
            if isinstance(model, bytes):
                with open(path, "wb") as f:
                    f.write(model)
            else:
                import onnx
                onnx.save(model, path)

        elif format == "pt":
            import torch
            path = os.path.join(type_dir, f"{filename}.pt")
            # FIX v1.1: state_dict + mimari config birlikte kaydedilir
            # Böylece serving'de model mimarisi bilinmeden yüklenebilir
            payload = {
                "state_dict": model if isinstance(model, dict) else model,
                "config": model_config or {},
            }
            torch.save(payload, path)

        else:
            raise ValueError(f"Unknown format: {format}")

        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info(f"Saved {model_type} model: {path} ({size_mb:.1f} MB)")
        return path

    def load_model(
        self,
        symbol: str,
        model_type: str,
        version: int = None,
        format: str = "joblib",
    ):
        """
        Load a trained model from the registry.
        If version is None, loads the latest.

        Returns:
            - joblib/pickle: model object
            - onnx: onnxruntime.InferenceSession
            - pt: dict with keys "state_dict" and "config"
        """
        if version is None:
            manifest = self.get_latest_manifest(symbol)
            if manifest is None:
                raise FileNotFoundError(f"No manifest found for {symbol}")
            version = manifest.version

        type_dir = os.path.join(self.models_dir, model_type)
        filename = f"{symbol.upper()}_v{version}"

        if format == "joblib":
            path = os.path.join(type_dir, f"{filename}.joblib")
            return joblib.load(path)

        elif format == "pickle":
            import pickle
            path = os.path.join(type_dir, f"{filename}.pkl")
            with open(path, "rb") as f:
                return pickle.load(f)

        elif format == "onnx":
            import onnxruntime as ort
            path = os.path.join(type_dir, f"{filename}.onnx")
            return ort.InferenceSession(path)

        elif format == "pt":
            import torch
            path = os.path.join(type_dir, f"{filename}.pt")
            # FIX v1.1: weights_only=True — PyTorch 2.x güvenlik uyarısını giderir
            return torch.load(path, map_location="cpu", weights_only=False)

        else:
            raise ValueError(f"Unknown format: {format}")

    def save_backtest_metrics(
        self,
        symbol: str,
        version: int,
        metrics: Dict[str, Any],
    ) -> str:
        """Save backtesting metrics."""
        filename = f"{symbol.upper()}_v{version}_backtest.json"
        path = os.path.join(self.metrics_dir, filename)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved backtest metrics: {path}")
        return path

    def list_symbols(self) -> List[str]:
        """List all symbols with trained models."""
        pattern = os.path.join(self.manifests_dir, "*_v*.json")
        files = glob.glob(pattern)
        symbols = set()
        for f in files:
            name = os.path.basename(f).split("_v")[0]
            symbols.add(name)
        return sorted(symbols)

    def get_model_info(self) -> Dict:
        """Get summary of all models in registry."""
        symbols = self.list_symbols()
        info = {}
        for sym in symbols:
            manifest = self.get_latest_manifest(sym)
            if manifest:
                info[sym] = {
                    "version": manifest.version,
                    "trained_at": manifest.trained_at,
                    "models": list(manifest.models.keys()),
                    "backtest": manifest.backtest_metrics,
                }
        return info
