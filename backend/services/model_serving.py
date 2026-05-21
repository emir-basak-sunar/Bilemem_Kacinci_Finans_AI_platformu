"""
FinAI Model Serving v2.0
=========================
Loads pre-trained models from the registry and runs inference.
Replaces the old "train-on-request" approach.

v2.0 Changes:
- Multi-manifest resolution: SECTOR → EQUITY_UNIVERSAL → CRYPTO → UNIVERSAL fallback
- Horizon-aware model selection (H5/H10/H20)
- symbol_encoding from manifest (deterministic, no hashing)
- Import path fix: training.features.feature_engineering

Supports:
- XGBoost/LightGBM/CatBoost via joblib
- LSTM/TCN/TFT via ONNX Runtime (framework-agnostic, fast)
- ARIMA via pickle
"""
import os
import sys
import logging
import time
import json
import glob
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also ensure the project root's parent is in path for 'training' package
# when backend/ and training/ are siblings under the same root
_WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

logger = logging.getLogger(__name__)

# ============================================================
# Sector mapping for manifest resolution
# ============================================================
SECTOR_MAP = {
    # Tech
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "META": "TECH",
    "NVDA": "TECH", "AMD": "TECH", "INTC": "TECH", "AVGO": "TECH",
    "CSCO": "TECH", "ORCL": "TECH", "CRM": "TECH", "ADBE": "TECH",
    "QCOM": "TECH",
    # Finance
    "JPM": "FINANCE", "BAC": "FINANCE", "GS": "FINANCE",
    "V": "FINANCE", "MA": "FINANCE",
    # Consumer
    "AMZN": "CONSUMER", "TSLA": "CONSUMER", "NFLX": "CONSUMER",
    "DIS": "CONSUMER", "UBER": "CONSUMER",
    # Fintech / Other
    "PYPL": "OTHER", "SQ": "OTHER", "COIN": "OTHER", "IBM": "OTHER",
    # Crypto
    "BTC-USD": "CRYPTO", "ETH-USD": "CRYPTO",
}

CRYPTO_SYMBOLS = frozenset({"BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
                             "XRP-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"})


class ModelServer:
    """
    Loads and serves pre-trained models.

    Manifest resolution chain:
    1. SECTOR_{sector}_v{version}  (most specific)
    2. EQUITY_UNIVERSAL_v{version} / CRYPTO_UNIVERSAL_v{version}
    3. UNIVERSAL_v{version}        (legacy fallback)

    Usage:
        server = ModelServer()
        result = server.predict("AAPL", market_data, model_type="ensemble", horizon=10)
    """

    def __init__(self, registry_dir: str = None):
        if registry_dir is None:
            _this_file = os.path.abspath(__file__)
            _services_dir = os.path.dirname(_this_file)
            _backend_dir = os.path.dirname(_services_dir)
            _workspace_dir = os.path.dirname(_backend_dir)
            # Önce production trained_models/ klasörüne bak, yoksa training/model_registry/ kullan
            _prod_registry = os.path.join(_workspace_dir, "trained_models")
            _dev_registry  = os.path.join(_workspace_dir, "training", "model_registry")
            _prod_manifests = os.path.join(_prod_registry, "manifests")
            if os.path.isdir(_prod_manifests) and len(glob.glob(os.path.join(_prod_manifests, "*.json"))) > 0:
                self.registry_dir = _prod_registry
            else:
                self.registry_dir = _dev_registry
            logger.info(f"ModelServer registry: {self.registry_dir}")
        else:
            self.registry_dir = registry_dir

        self.models_dir = os.path.join(self.registry_dir, "models")
        self.manifests_dir = os.path.join(self.registry_dir, "manifests")
        self.metrics_dir = os.path.join(self.registry_dir, "metrics")

        # Try to import ONNX Runtime
        try:
            import onnxruntime as ort
            self.ort = ort
            self.ort_options = ort.SessionOptions()
            self.ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_options.intra_op_num_threads = 4
            logger.info("ONNX Runtime available — DL models will use optimized inference")
        except ImportError:
            self.ort = None
            logger.warning("ONNX Runtime not available — DL models will fall back to PyTorch")

        self._cache: Dict[str, Any] = {}
        self._manifest_cache: Dict[str, dict] = {}

    # ============================================================
    # Manifest resolution
    # ============================================================
    def _load_manifest(self, symbol: str, version: int = None) -> Optional[dict]:
        """Load manifest for a symbol. If version given, load that specific version,
        otherwise load the latest."""
        if version is not None:
            cache_key = f"{symbol.upper()}_v{version}"
            if cache_key in self._manifest_cache:
                return self._manifest_cache[cache_key]

            path = os.path.join(self.manifests_dir, f"{symbol.upper()}_v{version}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                self._manifest_cache[cache_key] = data
                return data
            return None

        # Find latest version — FIX: numerik sort (v10 > v9 doğru çalışır)
        pattern = os.path.join(self.manifests_dir, f"{symbol.upper()}_v*.json")
        files = glob.glob(pattern)
        if not files:
            return None

        def _version_num(path: str) -> int:
            try:
                return int(os.path.basename(path).split("_v")[1].split(".")[0])
            except (ValueError, IndexError):
                return 0

        latest_file = max(files, key=_version_num)
        cache_key = os.path.basename(latest_file).replace(".json", "")
        if cache_key in self._manifest_cache:
            return self._manifest_cache[cache_key]

        with open(latest_file) as f:
            data = json.load(f)
        self._manifest_cache[cache_key] = data
        return data

    def _resolve_manifest(self, symbol: str, horizon: int = 1) -> Optional[dict]:
        """
        Smart manifest resolution chain:
        1. Horizon-specific (EQUITY_H5, EQUITY_H10, EQUITY_H20)
        2. Sector-specific (SECTOR_TECH, SECTOR_FINANCE, etc.)
        3. Asset-class universal (EQUITY_UNIVERSAL / CRYPTO_UNIVERSAL)
        4. Legacy fallback (UNIVERSAL)
        """
        symbol_upper = symbol.upper()
        is_crypto = symbol_upper in CRYPTO_SYMBOLS or symbol_upper.endswith("-USD")

        candidates = []

        if is_crypto:
            candidates.append("CRYPTO_UNIVERSAL")
        else:
            # Horizon-specific models
            if horizon >= 20:
                candidates.append("EQUITY_H20")
            elif horizon >= 10:
                candidates.append("EQUITY_H10")
            elif horizon >= 5:
                candidates.append("EQUITY_H5")

            # Sector model
            sector = SECTOR_MAP.get(symbol_upper)
            if sector:
                candidates.append(f"SECTOR_{sector}")

            # General equity
            candidates.append("EQUITY_UNIVERSAL")

        # Legacy fallback
        candidates.append("UNIVERSAL")

        for candidate in candidates:
            manifest = self._load_manifest(candidate, version=1)
            if manifest is not None:
                logger.debug(f"Resolved manifest for {symbol}: {candidate}")
                return manifest

        return None

    def _resolve_dl_manifest(self, symbol: str) -> Optional[dict]:
        """Resolve DL model manifest (v2/v3). EQUITY_UNIVERSAL → UNIVERSAL fallback."""
        symbol_upper = symbol.upper()
        is_crypto = symbol_upper in CRYPTO_SYMBOLS or symbol_upper.endswith("-USD")

        # v3 önce dene (daha yeni), sonra v2, sonra v1
        candidates = []
        if is_crypto:
            candidates += [("CRYPTO_UNIVERSAL", 3), ("CRYPTO_UNIVERSAL", 2), ("CRYPTO_UNIVERSAL", 1)]
        else:
            candidates += [
                ("EQUITY_UNIVERSAL", 3), ("EQUITY_UNIVERSAL", 2),
                ("UNIVERSAL", 3), ("UNIVERSAL", 2), ("UNIVERSAL", 1),
            ]

        for cand_sym, cand_ver in candidates:
            manifest = self._load_manifest(cand_sym, version=cand_ver)
            if manifest is not None:
                # DL model dosyası gerçekten var mı kontrol et
                for dl_type in ["lstm", "tcn", "tft"]:
                    model_dir = os.path.join(self.models_dir, dl_type)
                    prefix = f"{cand_sym.upper()}_v{cand_ver}"
                    if (os.path.exists(os.path.join(model_dir, f"{prefix}.onnx")) or
                        os.path.exists(os.path.join(model_dir, f"{prefix}.pt"))):
                        logger.debug(f"DL manifest resolved: {cand_sym}_v{cand_ver}")
                        return manifest
        return None

    # ============================================================
    # Model loading
    # ============================================================
    def _get_model(self, model_type: str, symbol: str, version: int):
        """Load model from cache or disk. ONNX → joblib → pt → pkl sırasıyla dener."""
        cache_key = f"{symbol}_{model_type}_v{version}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        model_dir = os.path.join(self.models_dir, model_type)
        prefix = f"{symbol.upper()}_v{version}"

        model = None

        # 1. ONNX (en hızlı inference)
        onnx_path = os.path.join(model_dir, f"{prefix}.onnx")
        if os.path.exists(onnx_path) and self.ort:
            model = self.ort.InferenceSession(onnx_path, self.ort_options)
            logger.info(f"Loaded ONNX model: {onnx_path}")

        # 2. joblib (tree modeller)
        if model is None:
            joblib_path = os.path.join(model_dir, f"{prefix}.joblib")
            if os.path.exists(joblib_path):
                model = joblib.load(joblib_path)
                logger.info(f"Loaded joblib model: {joblib_path}")

        # 3. PyTorch .pt (DL modeller — ONNX yoksa)
        if model is None:
            pt_path = os.path.join(model_dir, f"{prefix}.pt")
            if os.path.exists(pt_path):
                try:
                    import torch
                    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
                    model = payload  # dict: {"state_dict": ..., "config": ...}
                    logger.info(f"Loaded PyTorch model: {pt_path}")
                except Exception as e:
                    logger.warning(f"PyTorch load failed for {pt_path}: {e}")

        # 4. pickle (ARIMA/SARIMA/SARIMAX)
        if model is None:
            pkl_path = os.path.join(model_dir, f"{prefix}.pkl")
            if os.path.exists(pkl_path):
                import pickle
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"Loaded pickle model: {pkl_path}")

        if model is None:
            raise FileNotFoundError(f"No model found for {model_type}/{symbol}_v{version}")

        self._cache[cache_key] = model
        return model

    def _get_scaler(self, symbol: str, version: int):
        """Load StandardScaler for DL models."""
        cache_key = f"scaler_{symbol}_v{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        scaler_path = os.path.join(self.models_dir, f"scaler_{symbol}_v{version}.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            self._cache[cache_key] = scaler
            return scaler
        return None

    # ============================================================
    # Prediction
    # ============================================================
    def predict(
        self,
        symbol: str,
        market_data: list,
        model_type: str = "ensemble",
        horizon: int = 10,
    ) -> Dict:
        """
        Run prediction using pre-trained models.

        Args:
            symbol: Stock symbol
            market_data: List of OHLCV dicts from market data service
            model_type: ensemble, xgboost, lightgbm, catboost, lstm, tcn, tft, arima
            horizon: Number of future bars to predict

        Returns:
            Dict with prediction, models, future_path, confidence intervals
        """
        t0 = time.time()

        if not market_data or len(market_data) < 30:
            return {"error": "Not enough data for prediction (need >= 30 bars)"}

        # ── Resolve manifest (smart chain) ──
        manifest = self._resolve_manifest(symbol, horizon)
        if manifest is None:
            return {"error": f"No trained model found for {symbol}. Train models first."}

        model_symbol = manifest["symbol"]
        version = manifest["version"]
        feature_columns = manifest.get("feature_columns", [])
        ensemble_weights = manifest.get("ensemble_weights", {})
        symbol_encoding = manifest.get("symbol_encoding", {})

        # DL manifest (v2)
        dl_manifest = self._resolve_dl_manifest(symbol)
        dl_version = dl_manifest["version"] if dl_manifest else version

        # Build features from market data
        # Use the new import path (training.features.feature_engineering)
        try:
            from training.features.feature_engineering import build_features, get_feature_columns
        except ImportError:
            # Fallback to old import path
            from training.feature_engineering import build_features, get_feature_columns

        df = pd.DataFrame(market_data)
        # Sadece temel OHLCV sütunlarını build_features'a geç
        # market_data'da sma/ema/rsi gibi ekstra sütunlar olabilir — bunlar feature_engineering ile yeniden hesaplanır
        ohlcv_cols = [c for c in ['time', 'open', 'high', 'low', 'close', 'volume', 'symbol'] if c in df.columns]
        df_ohlcv = df[ohlcv_cols].copy()

        # FIX: drop_na=False — inference'ta 200-bar warmup drop'u tüm veriyi silebilir.
        # Özellikle period="1mo" → ~150 bar → sma_200 warmup sonrası 0 satır kalır.
        # Bunun yerine: NaN'ları 0 ile doldur, son satırı kullan.
        df_featured = build_features(df_ohlcv, drop_na=False)

        # Inf değerleri NaN'a çevir, sonra 0 ile doldur
        df_featured = df_featured.replace([float('inf'), float('-inf')], float('nan'))
        df_featured = df_featured.fillna(0)

        # En az close ve time sütunları dolu olmalı
        df_featured = df_featured[df_featured['close'].notna() & (df_featured['close'] > 0)]

        if df_featured.empty:
            return {"error": "Feature engineering produced no valid rows"}

        # Add symbol_encoded — use manifest encoding if available, else hash
        if "symbol_encoded" not in df_featured.columns:
            if symbol.upper() in symbol_encoding:
                df_featured["symbol_encoded"] = symbol_encoding[symbol.upper()]
            else:
                import hashlib
                symbol_code = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 1000
                df_featured["symbol_encoded"] = symbol_code

        # Build the exact feature set the model was trained on.
        # Missing features are filled with 0.
        if feature_columns:
            for col in feature_columns:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            aligned_features = feature_columns
        else:
            aligned_features = get_feature_columns(df_featured)

        last_row = df_featured.iloc[[-1]][aligned_features]
        current_price = float(df_featured.iloc[-1]["close"])
        last_time = int(df_featured.iloc[-1]["time"])

        # DL features — symbol_encoded dahil (DL modeller 128 feature ile eğitildi)
        dl_features = aligned_features  # symbol_encoded çıkarma

        # ── DL Models (LSTM/TCN/TFT) ──
        dl_model_symbol = dl_manifest["symbol"] if dl_manifest else model_symbol
        dl_version_num  = dl_manifest["version"] if dl_manifest else version
        scaler = self._get_scaler(dl_model_symbol, dl_version_num)
        seq_length = dl_manifest.get("seq_length", 60) if dl_manifest else 60

        # DL feature columns — manifest'ten al, yoksa aligned_features kullan
        dl_manifest_features = dl_manifest.get("feature_columns", []) if dl_manifest else []
        if dl_manifest_features:
            # Manifest'teki feature'ları kullan (eğitimde kullanılan tam liste)
            # Eksik feature'ları 0 ile doldur (makro, CS features — inference'ta mevcut olmayabilir)
            for col in dl_manifest_features:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            dl_feature_cols = dl_manifest_features
        else:
            dl_feature_cols = aligned_features

        predictions = {}
        confidence_intervals = None

        # ── Tree Models ──
        for tree_type in ["xgboost", "lightgbm", "catboost"]:
            if model_type not in ["ensemble", tree_type]:
                continue
            try:
                model = self._get_model(tree_type, model_symbol, version)

                if self.ort and hasattr(model, "run"):
                    # ONNX Runtime — numpy array gerektirir
                    input_data = last_row.values.astype(np.float32)
                    pred = model.run(None, {"features": input_data})[0]
                    predictions[tree_type] = float(pred[0])
                else:
                    # joblib model — eski XGBoost/LGBM/CatBoost numpy array ile daha güvenli
                    input_arr = last_row.values.astype(np.float64)
                    predictions[tree_type] = float(model.predict(input_arr)[0])

            except Exception as e:
                logger.warning(f"Failed to predict with {tree_type}: {e}")

        # ── DL Models (LSTM/TCN/TFT) ──
        dl_model_symbol = dl_manifest["symbol"] if dl_manifest else model_symbol
        _dl_ver         = dl_manifest["version"] if dl_manifest else version
        scaler          = self._get_scaler(dl_model_symbol, _dl_ver)
        seq_length      = dl_manifest.get("seq_length", 60) if dl_manifest else 60

        # DL feature columns — manifest'ten al, eksikleri 0 ile doldur
        _dl_feat_list = dl_manifest.get("feature_columns", []) if dl_manifest else []
        if _dl_feat_list:
            for col in _dl_feat_list:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            dl_feature_cols = _dl_feat_list
        else:
            dl_feature_cols = aligned_features

        for dl_type in ["lstm", "tcn", "tft"]:
            if model_type not in ["ensemble", dl_type]:
                continue
            try:
                model = self._get_model(dl_type, dl_model_symbol, _dl_ver)

                # Prepare DL features — ensure columns exist
                for col in dl_feature_cols:
                    if col not in df_featured.columns:
                        df_featured[col] = 0.0

                seq_data = df_featured[dl_feature_cols].tail(seq_length).values.astype(np.float32)

                # Scaler uygula (varsa)
                if scaler:
                    try:
                        seq_data = scaler.transform(seq_data).astype(np.float32)
                    except Exception as e:
                        logger.warning(f"Scaler transform failed for {dl_type}: {e}")

                # Pad if needed
                actual_len = seq_data.shape[0]
                if actual_len < seq_length:
                    pad = np.zeros((seq_length - actual_len, seq_data.shape[1]), dtype=np.float32)
                    seq_data = np.vstack([pad, seq_data])
                input_data = seq_data.reshape(1, seq_length, -1).astype(np.float32)

                if self.ort and hasattr(model, "run"):
                    # ONNX Runtime inference
                    output = model.run(None, {"input": input_data})[0]
                    if dl_type == "tft" and output.shape[-1] > 1:
                        quantiles = output[0]
                        predictions[dl_type] = float(quantiles[2])
                        confidence_intervals = {
                            "p10": float(quantiles[0]),
                            "p25": float(quantiles[1]),
                            "p50": float(quantiles[2]),
                            "p75": float(quantiles[3]),
                            "p90": float(quantiles[4]),
                        }
                    else:
                        predictions[dl_type] = float(output[0])

                elif isinstance(model, dict) and "state_dict" in model:
                    # PyTorch .pt payload — architectures.py'den model oluştur
                    try:
                        import torch
                        from training.models.dl.architectures import BiLSTM, TCN, TFTModel
                        cfg = model.get("config", {})
                        n_features = input_data.shape[2]

                        if dl_type == "lstm":
                            net = BiLSTM(
                                input_size=n_features,
                                hidden_size=cfg.get("hidden_size", 256),
                                num_layers=cfg.get("num_layers", 3),
                                dropout=cfg.get("dropout", 0.2),
                                bidirectional=cfg.get("bidirectional", True),
                            )
                        elif dl_type == "tcn":
                            net = TCN(
                                input_size=n_features,
                                num_channels=cfg.get("num_channels", [64,64,128,128,256,256,128,64]),
                                kernel_size=cfg.get("kernel_size", 3),
                                dropout=cfg.get("dropout", 0.15),
                            )
                        else:  # tft
                            net = TFTModel(
                                input_size=n_features,
                                hidden_size=cfg.get("hidden_size", 128),
                                n_heads=cfg.get("attention_head_size", 8),
                                dropout=cfg.get("dropout", 0.1),
                            )

                        net.load_state_dict(model["state_dict"])
                        net.eval()
                        with torch.no_grad():
                            x = torch.tensor(input_data)
                            out = net(x)
                            if dl_type == "tft" and out.dim() > 1 and out.shape[-1] > 1:
                                predictions[dl_type] = float(out[0, 2].item())  # P50
                                confidence_intervals = {
                                    "p10": float(out[0, 0].item()),
                                    "p25": float(out[0, 1].item()),
                                    "p50": float(out[0, 2].item()),
                                    "p75": float(out[0, 3].item()),
                                    "p90": float(out[0, 4].item()),
                                }
                            else:
                                predictions[dl_type] = float(out[0].item())
                    except Exception as pt_e:
                        logger.warning(f"PyTorch inference failed for {dl_type}: {pt_e}")
                else:
                    logger.warning(f"Skipping {dl_type} — unsupported model format")

            except Exception as e:
                logger.warning(f"Failed to predict with {dl_type}: {e}")

        # ── ARIMA / SARIMA / SARIMAX ──
        for stat_type in ["arima", "sarima", "sarimax"]:
            if model_type not in ["ensemble", stat_type]:
                continue
            try:
                sym_upper = symbol.upper()
                stat_version = 1
                try:
                    stat_model = self._get_model(stat_type, sym_upper, stat_version)
                except FileNotFoundError:
                    try:
                        stat_model = self._get_model(stat_type, model_symbol, stat_version)
                    except FileNotFoundError:
                        logger.debug(f"No {stat_type} model for {sym_upper}")
                        continue

                # pmdarima / statsmodels predict
                try:
                    forecast = stat_model.predict(n_periods=horizon)
                    raw_pred = float(forecast[0]) if hasattr(forecast, '__len__') else float(forecast)
                except AttributeError:
                    # statsmodels SARIMAX — farklı API
                    forecast = stat_model.forecast(steps=horizon)
                    raw_pred = float(forecast.iloc[0]) if hasattr(forecast, 'iloc') else float(forecast[0])

                # ARIMA/SARIMA fiyat tahmini döndürür, return'e çevir
                if raw_pred > 10:  # fiyat değeri (return değil)
                    raw_pred = (raw_pred / current_price) - 1.0

                predictions[stat_type] = raw_pred
                logger.info(f"{stat_type} prediction for {sym_upper}: {raw_pred:.6f}")
            except Exception as e:
                logger.warning(f"{stat_type} prediction failed: {e}")

        if not predictions:
            return {"error": "No models produced predictions"}

        # ── Ensemble ──
        if model_type == "ensemble" and len(predictions) > 1:
            if ensemble_weights:
                weighted_sum = sum(predictions.get(k, 0) * w for k, w in ensemble_weights.items() if k in predictions)
                total_weight = sum(w for k, w in ensemble_weights.items() if k in predictions)
                prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
            else:
                prediction = np.mean(list(predictions.values()))
        else:
            prediction = predictions.get(model_type, list(predictions.values())[0])

        # ── Generate future path ──
        future_path = self._generate_future_path(
            df_featured, aligned_features, predictions,
            model_symbol, version, horizon, last_time, current_price,
            ensemble_prediction=float(prediction)
        )

        elapsed = round((time.time() - t0) * 1000, 1)

        result = {
            "current_price": current_price,
            "prediction": float(prediction),
            "models": {k: round(v, 6) for k, v in predictions.items()},
            "future_path": future_path,
            "confidence_intervals": confidence_intervals,
            "model_version": version,
            "manifest_used": model_symbol,
            "training_date": manifest.get("trained_at", "unknown"),
            "training_metrics": manifest.get("backtest_metrics", {}),
            "_response_time_ms": elapsed,
        }

        logger.info(f"[PREDICTION] {symbol}/{model_type} (manifest={model_symbol}) — {elapsed}ms, "
                     f"current={current_price:.2f}, pred={prediction:.6f}")

        return result

    def _generate_future_path(
        self, df_featured, feature_cols, predictions,
        model_symbol, version, horizon, last_time, current_price,
        ensemble_prediction: float = None
    ) -> List[Dict]:
        """
        Generate future price path.
        Step 1 is anchored to the ensemble prediction.
        Steps 2+ use XGBoost recursive prediction.
        """
        future_path = []

        try:
            model = self._get_model("xgboost", model_symbol, version)
            current_input = df_featured.iloc[[-1]][feature_cols].copy()
            # FIX: numpy array kullan — eski XGBoost joblib modelleri DataFrame dtype sorununu önler
            current_input_arr = current_input.values.astype(np.float64)

            times = df_featured["time"].values
            avg_step = int(np.median(np.diff(times[-20:]))) if len(times) >= 2 else 86400

            if ensemble_prediction is not None:
                first_price = round(current_price * (1 + ensemble_prediction), 4)
            else:
                pred_return = float(model.predict(current_input_arr)[0])
                first_price = round(current_price * (1 + pred_return), 4)

            future_path.append({"time": last_time + avg_step, "value": first_price})
            curr_price = first_price

            for i in range(2, horizon + 1):
                pred_return = float(model.predict(current_input_arr)[0])
                next_price = round(curr_price * (1 + pred_return), 4)
                future_path.append({"time": last_time + (i * avg_step), "value": next_price})
                curr_price = next_price

        except Exception as e:
            logger.warning(f"Future path generation failed: {e}")
            base_return = ensemble_prediction if ensemble_prediction is not None else np.mean(list(predictions.values()))
            avg_step = 3600
            for i in range(1, horizon + 1):
                future_path.append({
                    "time": last_time + (i * avg_step),
                    "value": round(current_price * (1 + base_return * i), 4),
                })

        return future_path

    # ============================================================
    # Registry info
    # ============================================================
    def get_available_models(self) -> Dict:
        """List all available models in registry."""
        result = {}
        pattern = os.path.join(self.manifests_dir, "*_v*.json")

        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                manifest = json.load(fh)
            sym = manifest["symbol"]
            ver = manifest["version"]
            key = f"{sym}_v{ver}"
            result[key] = {
                "symbol": sym,
                "version": ver,
                "trained_at": manifest.get("trained_at", "unknown"),
                "models": list(manifest.get("models", {}).keys()),
                "backtest_metrics": manifest.get("backtest_metrics", {}),
            }

        return result

    def get_all_metrics(self) -> Dict:
        """Load all backtest and SHAP metrics from the metrics directory."""
        result = {"backtest": {}, "shap": {}}

        if not os.path.exists(self.metrics_dir):
            return result

        for f in sorted(os.listdir(self.metrics_dir)):
            if not f.endswith(".json"):
                continue
            path = os.path.join(self.metrics_dir, f)
            with open(path) as fh:
                data = json.load(fh)

            name = f.replace(".json", "")
            if "_backtest" in name:
                result["backtest"][name] = data
            elif "_shap" in name:
                result["shap"][name] = data

        return result

    def get_all_manifests(self) -> List[Dict]:
        """Load all manifests with full details."""
        manifests = []
        pattern = os.path.join(self.manifests_dir, "*_v*.json")

        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                manifest = json.load(fh)
            manifest["_filename"] = os.path.basename(f)
            manifests.append(manifest)

        return manifests

    def get_training_scores(self) -> Dict:
        """
        Aggregated training scores summary — used by MLflow and frontend.
        Returns per-model-family metrics from the latest Colab training.
        """
        metrics = self.get_all_metrics()
        backtest = metrics.get("backtest", {})

        summary = {}
        for name, data in backtest.items():
            # e.g. EQUITY_UNIVERSAL_v1_backtest → EQUITY_UNIVERSAL_v1
            clean_name = name.replace("_backtest", "")
            if isinstance(data, dict):
                summary[clean_name] = {}
                for model_name, model_metrics in data.items():
                    if isinstance(model_metrics, dict):
                        summary[clean_name][model_name] = {
                            k: v for k, v in model_metrics.items()
                            if k in ("mae", "rmse", "smape", "directional_accuracy",
                                     "ic", "r2", "sharpe", "sortino", "max_drawdown",
                                     "annual_return", "n_samples")
                        }

        return summary


# Singleton instance
_server: Optional[ModelServer] = None

def get_model_server() -> ModelServer:
    global _server
    if _server is None:
        _server = ModelServer()
    return _server
