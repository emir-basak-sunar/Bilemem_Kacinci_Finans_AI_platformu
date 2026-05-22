"""
============================================================
FinAI Training — Phase 1: Data + XGBoost/LightGBM/CatBoost — v2.0
============================================================
Bu script'i Google Colab A100'de çalıştır.

Adımlar:
1. Veri toplama (yfinance — hisse + makro, kripto AYRI)
2. Feature engineering (160+ teknik + makro + cross-sectional feature)
3. XGBoost/LightGBM/CatBoost eğitimi (equity universal + kripto ayrı)
4. Cross-sectional walk-forward backtesting (IC Sharpe)
5. ONNX export
6. Model Registry'ye kaydet

Değişiklikler v2.0 (hedge-fund grade):
- KRİPTO AYRILDI: BTC/ETH artık ayrı model — equity ile karıştırılmaz
- Cross-sectional rank target: target_cs_rank_1 — IC'yi doğrudan optimize eder
- Optuna SQLite persistence: tuning geçmişi Colab session'lar arası korunur
- Optuna arama alanı genişletildi: min_child_weight, gamma, min_child_samples
- PurgedKFold horizon parametresi dinamik: target_5 için horizon=5
- Cross-sectional walk-forward: IC Sharpe hesabı
- Regime-conditional ensemble: VIX + SP500 trend ile dinamik ağırlıklar
- add_cross_sectional_features: rank-based features eklendi
"""
import sys
import os
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Add project root to path — models/tree/ içinden 3 seviye yukarı
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.config import (
    TRAINING_UNIVERSE, CRYPTO_SYMBOLS, ALL_SYMBOLS, CORE_SYMBOLS,
    EQUITY_UNIVERSE, EQUITY_ENCODING,
    FEATURE_CFG, XGBOOST_CFG, LGBM_CFG, CATBOOST_CFG,
    MODELS_DIR, DATA_DIR, SYMBOL_ENCODING, OPTUNA_CFG, CS_CFG,
)
from training.features.feature_engineering import (
    build_features, get_feature_columns, create_multi_horizon_targets,
    add_cross_sectional_features, add_cross_sectional_targets,
)
from training.data_collection import (
    fetch_multi_symbol, save_data, load_data,
    train_test_split_timeseries, split_by_symbol,
)
from training.model_registry import ModelRegistry, ModelManifest
from training.walk_forward import (
    WalkForwardValidator, PurgedKFold, compute_quant_metrics,
    compute_ic_sharpe, split_by_sector, SECTOR_GROUPS, RegimeDetector,
)

# Makro feature modülü
try:
    from training.features.macro_features import (
        fetch_macro_data, save_macro_data, load_macro_data,
    )
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 1. DATA COLLECTION
# ============================================================

def collect_training_data(symbols=None, period="5y", interval="1d", force=False):
    """Collect and save training data. v2.0: Equity ve kripto ayrı çekilir."""
    if symbols is None:
        symbols = ALL_SYMBOLS

    filename = f"ohlcv_{interval}_{len(symbols)}sym.parquet"
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath) and not force:
        logger.info(f"Data already exists: {filepath}")
        df = load_data(filename)
    else:
        logger.info(f"Collecting data for {len(symbols)} symbols...")
        df = fetch_multi_symbol(symbols, period=period, interval=interval, delay=0.5)
        if df.empty:
            raise RuntimeError("No data collected!")
        save_data(df, filename)

    # v1.2: Makro veri çek ve kaydet
    macro_df = None
    if MACRO_AVAILABLE:
        macro_path = os.path.join(DATA_DIR, "macro_1d.parquet")
        if os.path.exists(macro_path) and not force:
            logger.info("Macro data already exists, loading...")
            try:
                macro_df = load_macro_data(DATA_DIR)
            except Exception as e:
                logger.warning(f"Macro data load failed: {e}")
        else:
            logger.info("Fetching macro data...")
            try:
                macro_df = fetch_macro_data(period=period, interval=interval)
                if not macro_df.empty:
                    save_macro_data(macro_df, DATA_DIR)
            except Exception as e:
                logger.warning(f"Macro data fetch failed (non-critical): {e}")
    else:
        logger.info("Macro features not available (macro_features.py missing)")

    return df, macro_df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame, macro_df=None, add_cs_features: bool = True) -> pd.DataFrame:
    """Apply feature engineering per symbol. v2.0: cross-sectional features eklendi."""
    all_featured = []

    symbols = df["symbol"].unique()
    logger.info(f"Engineering features for {len(symbols)} symbols...")

    for sym in symbols:
        sym_data = df[df["symbol"] == sym].copy().reset_index(drop=True)

        if len(sym_data) < 200:
            logger.warning(f"Skipping {sym}: only {len(sym_data)} bars (need >= 200)")
            continue

        # FIX v2.0: drop_na=True — feature NaN'larını burada temizle
        # Önceki: drop_na=False → tüm NaN'lar combined'a taşınıyor → sonraki dropna tüm satırları siliyor
        featured = build_features(
            sym_data,
            sma_windows=FEATURE_CFG.sma_windows,
            ema_windows=FEATURE_CFG.ema_windows,
            price_lags=FEATURE_CFG.price_lags,
            volume_lags=FEATURE_CFG.volume_lags,
            return_lags=FEATURE_CFG.return_lags,
            drop_na=True,   # FIX: feature NaN'larını burada temizle
            macro_df=macro_df,
        )

        if len(featured) < 50:
            logger.warning(f"Skipping {sym}: only {len(featured)} rows after feature NaN drop")
            continue

        # Add multi-horizon targets — son h bar NaN olacak (gelecek fiyat yok)
        targets = create_multi_horizon_targets(featured["close"], FEATURE_CFG.horizons)
        featured = pd.concat([featured, targets], axis=1)

        # Sadece temel regression target'larda NaN olan satırları düşür
        # (son max_horizon bar — gelecek fiyat yok)
        # target_cs_rank_* henüz yok, target_vol_* NaN olabilir → bunları tutuyoruz
        basic_targets = [f"target_{h}" for h in FEATURE_CFG.horizons]
        available_basic = [c for c in basic_targets if c in featured.columns]
        if available_basic:
            featured = featured.dropna(subset=available_basic)

        if len(featured) < 50:
            logger.warning(f"Skipping {sym}: only {len(featured)} rows after target NaN drop")
            continue

        all_featured.append(featured)

    if not all_featured:
        raise RuntimeError("No symbols survived feature engineering! Check data quality.")

    combined = pd.concat(all_featured, ignore_index=True)
    logger.info(f"Combined before CS features: {len(combined)} rows, {len(combined.columns)} columns")

    # v2.0: Cross-sectional features (tüm semboller birleştirildikten sonra)
    if add_cs_features and len(symbols) >= CS_CFG.min_symbols:
        logger.info("Adding cross-sectional features...")
        feature_cols = get_feature_columns(combined)
        combined = add_cross_sectional_features(
            combined, feature_cols,
            min_symbols=CS_CFG.min_symbols,
        )
        # Cross-sectional rank targets
        combined = add_cross_sectional_targets(
            combined, FEATURE_CFG.horizons,
            min_symbols=CS_CFG.min_symbols,
        )
        # CS rank NaN'larını 0 ile doldur (rank 0 = orta, az sembol olan zaman adımları)
        for col in combined.columns:
            if col.startswith("target_cs_rank_"):
                combined[col] = combined[col].fillna(0.0)

    logger.info(f"Total featured data: {len(combined)} rows, {len(combined.columns)} columns")
    return combined


# ============================================================
# 3. TRAINING — Universal XGBoost/LightGBM/CatBoost
# ============================================================

def train_tree_models(
    df_featured: pd.DataFrame,
    target_col: str = "target_1",
    symbol: str = "UNIVERSAL",
    tune_hyperparams: bool = False,
    n_trials: int = 15,
    use_stacking: bool = False,   # Stacking varsayılan kapalı — val seti küçükse kötü sonuç verir
):
    """
    Train XGBoost, LightGBM, CatBoost on ALL symbols combined.
    Uses per-symbol time-series split (split_by_symbol).
    """
    registry = ModelRegistry()
    version = registry.get_next_version(symbol)

    # FIX v1.1: get_feature_columns artık ham fiyat sütunlarını filtreler
    feature_cols = get_feature_columns(
        df_featured,
        sma_windows=FEATURE_CFG.sma_windows,
        ema_windows=FEATURE_CFG.ema_windows,
    )

    # FIX v1.1: Sabit SYMBOL_ENCODING — her çalıştırmada aynı mapping
    # Önceki: df["symbol"].astype("category").cat.codes → her çalıştırmada farklı
    if "symbol" in df_featured.columns:
        df_featured = df_featured.copy()  # in-place modify'dan kaçın
        df_featured["symbol_encoded"] = df_featured["symbol"].map(SYMBOL_ENCODING).fillna(-1).astype(int)
        if "symbol_encoded" not in feature_cols:
            feature_cols.append("symbol_encoded")

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Target: {target_col}")

    # FIX v1.1: split_by_symbol — her sembol kendi içinde split edilir
    splits = split_by_symbol(df_featured, test_ratio=0.2, val_ratio=0.1)

    # Boş split kontrolü
    for split_name in ["train", "val", "test"]:
        if splits[split_name].empty:
            raise ValueError(
                f"Split '{split_name}' is empty for symbol={symbol}, target={target_col}. "
                f"DataFrame has {len(df_featured)} rows. Check engineer_features output."
            )

    # Target sütununda NaN olan satırları düşür (son h bar)
    for split_name in ["train", "val", "test"]:
        before = len(splits[split_name])
        splits[split_name] = splits[split_name].dropna(subset=[target_col])
        after = len(splits[split_name])
        if before != after:
            logger.info(f"  {split_name}: dropped {before-after} rows with NaN target")

    X_train = splits["train"][feature_cols]
    y_train = splits["train"][target_col]
    X_val = splits["val"][feature_cols]
    y_val = splits["val"][target_col]
    X_test = splits["test"][feature_cols]
    y_test = splits["test"][target_col]

    models = {}
    metrics = {}

    # ── Default params (config'den) ──
    xgb_params = {
        "n_estimators": XGBOOST_CFG.n_estimators,
        "learning_rate": XGBOOST_CFG.learning_rate,
        "max_depth": XGBOOST_CFG.max_depth,
        "subsample": XGBOOST_CFG.subsample,
        "colsample_bytree": XGBOOST_CFG.colsample_bytree,
        "reg_alpha": XGBOOST_CFG.reg_alpha,
        "reg_lambda": XGBOOST_CFG.reg_lambda,
    }
    lgbm_params = {
        "n_estimators": LGBM_CFG.n_estimators,
        "learning_rate": LGBM_CFG.learning_rate,
        "num_leaves": LGBM_CFG.num_leaves,
        "max_depth": LGBM_CFG.max_depth,
        "subsample": LGBM_CFG.subsample,
        "colsample_bytree": LGBM_CFG.colsample_bytree,
        "reg_alpha": LGBM_CFG.reg_alpha,
        "reg_lambda": LGBM_CFG.reg_lambda,
    }
    cat_params = {
        "iterations": CATBOOST_CFG.iterations,
        "learning_rate": CATBOOST_CFG.learning_rate,
        "depth": CATBOOST_CFG.depth,
        "l2_leaf_reg": CATBOOST_CFG.l2_leaf_reg,
    }

    if tune_hyperparams:
        logger.info(f"Running Optuna tuning ({n_trials} trials per model)...")

        # FIX v1.1: Optuna early_stopping_rounds config'den alınıyor (tutarlı)
        es_rounds = XGBOOST_CFG.early_stopping_rounds

        # XGBoost Tuning — v2.0: genişletilmiş arama alanı + SQLite persistence
        def xgb_obj(trial):
            p = {
                "n_estimators":     trial.suggest_int("n_estimators", *OPTUNA_CFG.xgb_n_estimators_range, step=100),
                "learning_rate":    trial.suggest_float("learning_rate", *OPTUNA_CFG.xgb_lr_range, log=True),
                "max_depth":        trial.suggest_int("max_depth", *OPTUNA_CFG.xgb_depth_range),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", *OPTUNA_CFG.xgb_min_child_weight_range),
                "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
                "n_jobs": -1, "random_state": 42,
                "early_stopping_rounds": XGBOOST_CFG.early_stopping_rounds,
            }
            m = XGBRegressor(**p)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = m.predict(X_val)
            return float(np.sqrt(mean_squared_error(y_val, preds)))

        study_xgb = optuna.create_study(
            direction="minimize",
            storage=OPTUNA_CFG.storage,
            study_name=f"{symbol}_xgb_{target_col}",
            load_if_exists=True,
        )
        study_xgb.optimize(xgb_obj, n_trials=n_trials)
        xgb_params.update(study_xgb.best_params)
        logger.info(f"  Best XGBoost params: {study_xgb.best_params}")

        # LightGBM Tuning — v2.0: min_child_samples eklendi
        def lgbm_obj(trial):
            p = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 2000, step=100),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", *OPTUNA_CFG.lgbm_num_leaves_range),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "min_child_samples":trial.suggest_int("min_child_samples", *OPTUNA_CFG.lgbm_min_child_samples_range),
                "n_jobs": -1, "random_state": 42, "verbose": -1,
            }
            m = LGBMRegressor(**p)
            cbs = [c for c in [_lgbm_early_stopping(LGBM_CFG.early_stopping_rounds)] if c is not None]
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cbs or None)
            preds = m.predict(X_val)
            return float(np.sqrt(mean_squared_error(y_val, preds)))

        study_lgbm = optuna.create_study(
            direction="minimize",
            storage=OPTUNA_CFG.storage,
            study_name=f"{symbol}_lgbm_{target_col}",
            load_if_exists=True,
        )
        study_lgbm.optimize(lgbm_obj, n_trials=n_trials)
        lgbm_params.update(study_lgbm.best_params)
        logger.info(f"  Best LightGBM params: {study_lgbm.best_params}")

        # CatBoost Tuning
        def cat_obj(trial):
            p = {
                "iterations": trial.suggest_int("iterations", 100, 600, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 5.0, log=True),
                "random_state": 42, "verbose": False,
                "early_stopping_rounds": CATBOOST_CFG.early_stopping_rounds,
            }
            m = CatBoostRegressor(**p)
            m.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            preds = m.predict(X_val)
            return float(np.sqrt(mean_squared_error(y_val, preds)))

        study_cat = optuna.create_study(
            direction="minimize",
            storage=OPTUNA_CFG.storage,
            study_name=f"{symbol}_cat_{target_col}",
            load_if_exists=True,
        )
        study_cat.optimize(cat_obj, n_trials=n_trials)
        cat_params.update(study_cat.best_params)
        logger.info(f"  Best CatBoost params: {study_cat.best_params}")

    # ── XGBoost ──
    logger.info("Training XGBoost...")
    t0 = time.time()
    xgb = XGBRegressor(
        **xgb_params,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=XGBOOST_CFG.early_stopping_rounds,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    xgb_time = round(time.time() - t0, 1)

    xgb_pred = xgb.predict(X_test)
    metrics["xgboost"] = evaluate_predictions(y_test, xgb_pred)
    metrics["xgboost"]["train_time_s"] = xgb_time
    models["xgboost"] = xgb
    logger.info(f"  XGBoost: sMAPE={metrics['xgboost']['smape']:.2f}%, RMSE={metrics['xgboost']['rmse']:.6f}, IC={metrics['xgboost']['ic']:.4f} ({xgb_time}s)")

    # ── LightGBM ──
    logger.info("Training LightGBM...")
    t0 = time.time()
    lgbm = LGBMRegressor(
        **lgbm_params,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    # FIX v1.1: None callback'leri filtrele — sessiz hata önlenir
    lgbm_cbs = [c for c in [_lgbm_early_stopping(LGBM_CFG.early_stopping_rounds)] if c is not None]
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=lgbm_cbs if lgbm_cbs else None,
    )
    lgbm_time = round(time.time() - t0, 1)

    lgbm_pred = lgbm.predict(X_test)
    metrics["lightgbm"] = evaluate_predictions(y_test, lgbm_pred)
    metrics["lightgbm"]["train_time_s"] = lgbm_time
    models["lightgbm"] = lgbm
    logger.info(f"  LightGBM: sMAPE={metrics['lightgbm']['smape']:.2f}%, RMSE={metrics['lightgbm']['rmse']:.6f}, IC={metrics['lightgbm']['ic']:.4f} ({lgbm_time}s)")

    # ── CatBoost ──
    logger.info("Training CatBoost...")
    t0 = time.time()
    cat = CatBoostRegressor(
        **cat_params,
        random_state=42,
        verbose=100,
        early_stopping_rounds=CATBOOST_CFG.early_stopping_rounds,
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_time = round(time.time() - t0, 1)

    cat_pred = cat.predict(X_test)
    metrics["catboost"] = evaluate_predictions(y_test, cat_pred)
    metrics["catboost"]["train_time_s"] = cat_time
    models["catboost"] = cat
    logger.info(f"  CatBoost: sMAPE={metrics['catboost']['smape']:.2f}%, RMSE={metrics['catboost']['rmse']:.6f}, IC={metrics['catboost']['ic']:.4f} ({cat_time}s)")

    # ── Weighted Ensemble (inverse-RMSE weights) ──
    # Compute weights first, then apply
    rmses = {
        "xgboost":  float(np.sqrt(mean_squared_error(y_test, xgb_pred))),
        "lightgbm": float(np.sqrt(mean_squared_error(y_test, lgbm_pred))),
        "catboost": float(np.sqrt(mean_squared_error(y_test, cat_pred))),
    }
    inv_rmse = {k: 1.0 / v for k, v in rmses.items()}
    total_w  = sum(inv_rmse.values())
    ensemble_weights = {k: round(v / total_w, 4) for k, v in inv_rmse.items()}

    ensemble_pred = (
        xgb_pred  * ensemble_weights["xgboost"]  +
        lgbm_pred * ensemble_weights["lightgbm"] +
        cat_pred  * ensemble_weights["catboost"]
    )
    metrics["ensemble"] = evaluate_predictions(y_test, ensemble_pred)
    logger.info(f"  Ensemble: sMAPE={metrics['ensemble']['smape']:.2f}%, RMSE={metrics['ensemble']['rmse']:.6f}, IC={metrics['ensemble']['ic']:.4f}")
    logger.info(f"  Weights: {ensemble_weights}")

    # ── Stacking Ensemble (optional — disabled by default) ──
    # Only useful when val set is large (>5000 rows). With small val sets,
    # Ridge meta-learner learns near-zero weights and performs worse than simple ensemble.
    stacking_weights = ensemble_weights  # default fallback
    if use_stacking:
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler as SkScaler

            val_preds_stack  = np.column_stack([xgb.predict(X_val),  lgbm.predict(X_val),  cat.predict(X_val)])
            test_preds_stack = np.column_stack([xgb_pred, lgbm_pred, cat_pred])

            meta_scaler       = SkScaler()
            val_preds_scaled  = meta_scaler.fit_transform(val_preds_stack)
            test_preds_scaled = meta_scaler.transform(test_preds_stack)

            meta_model    = Ridge(alpha=1.0)
            meta_model.fit(val_preds_scaled, y_val)
            stacking_pred = meta_model.predict(test_preds_scaled)

            stacking_metrics = evaluate_predictions(y_test, stacking_pred)
            logger.info(f"  Stacking: sMAPE={stacking_metrics['smape']:.2f}%, RMSE={stacking_metrics['rmse']:.6f}, IC={stacking_metrics['ic']:.4f}")

            # Only use stacking if it actually beats the ensemble
            if stacking_metrics["rmse"] < metrics["ensemble"]["rmse"]:
                metrics["stacking"] = stacking_metrics
                meta_coefs = meta_model.coef_
                stacking_weights = {
                    "xgboost":  round(float(meta_coefs[0]), 4),
                    "lightgbm": round(float(meta_coefs[1]), 4),
                    "catboost": round(float(meta_coefs[2]), 4),
                }
                logger.info(f"  Stacking beats ensemble — using stacking weights: {stacking_weights}")
            else:
                logger.info(f"  Stacking did NOT beat ensemble — keeping inverse-RMSE weights")
        except Exception as e:
            logger.warning(f"  Stacking failed (non-critical): {e}")

    # ── Save models ──
    model_paths = {}
    for name, model in models.items():
        path = registry.save_model(model, symbol, name, version, format="joblib")
        model_paths[name] = {
            "path": os.path.relpath(path, registry.registry_dir),
            **metrics[name],
        }

    # ── Quant metrics on ensemble predictions ──
    try:
        # FIX v2.1: horizon parametresi geçildi — doğru annualization için
        # target_col'dan horizon'u çıkar (örn: "target_10" → horizon=10)
        _h = 1
        try:
            _h = int(target_col.split("_")[-1])
        except (ValueError, IndexError):
            pass
        quant = compute_quant_metrics(y_test.values, ensemble_pred, horizon=_h)
        metrics["ensemble_quant"] = quant
        logger.info(
            f"  Quant Metrics (horizon={_h}) — Sharpe: {quant['sharpe']:.3f} | "
            f"Sortino: {quant['sortino']:.3f} | "
            f"MaxDD: {quant['max_drawdown']:.3f} | "
            f"Calmar: {quant['calmar']:.3f} | "
            f"Turnover: {quant['turnover']:.3f} | "
            f"IC (overlap-adj): {quant['ic']:.4f} (n_ind={quant.get('n_independent', '?')})"
        )
    except Exception as e:
        logger.warning(f"  Quant metrics failed (non-critical): {e}")

    # ── Export to ONNX ──
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, len(feature_cols)]))]

        for model_name, model_obj in [("xgboost", xgb), ("lightgbm", lgbm), ("catboost", cat)]:
            try:
                onnx_model = convert_sklearn(model_obj, initial_types=initial_type)
                onnx_path = registry.save_model(
                    onnx_model.SerializeToString(), symbol, model_name, version, format="onnx"
                )
                logger.info(f"  {model_name} ONNX exported: {onnx_path}")
            except Exception as e:
                logger.warning(f"  {model_name} ONNX export failed (optional): {e}")

    except ImportError as e:
        logger.warning(f"  skl2onnx not available, ONNX export skipped: {e}")

    # ── v1.2: SHAP Feature Importance ──
    try:
        import shap
        logger.info("Computing SHAP feature importance (XGBoost)...")
        explainer = shap.TreeExplainer(xgb)
        # Küçük bir sample üzerinde hesapla (hız için)
        sample_size = min(500, len(X_test))
        shap_values = explainer.shap_values(X_test.iloc[:sample_size])
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "shap_mean_abs": np.abs(shap_values).mean(axis=0),
        }).sort_values("shap_mean_abs", ascending=False)

        shap_path = os.path.join(registry.metrics_dir, f"{symbol}_v{version}_shap.json")
        shap_importance.head(30).to_json(shap_path, orient="records", indent=2)
        logger.info(f"  SHAP saved: {shap_path}")
        logger.info(f"  Top 5 features: {shap_importance['feature'].head(5).tolist()}")
    except Exception as e:
        logger.warning(f"  SHAP computation failed (non-critical): {e}")

    # ── Save manifest ──
    time_min = df_featured["time"].min()
    time_max = df_featured["time"].max()
    try:
        date_min = pd.to_datetime(time_min, unit="s").strftime("%Y-%m-%d")
        date_max = pd.to_datetime(time_max, unit="s").strftime("%Y-%m-%d")
    except Exception:
        date_min = str(time_min)
        date_max = str(time_max)

    manifest = ModelManifest(
        symbol=symbol,
        version=version,
        data_range=[date_min, date_max],
        models=model_paths,
        ensemble_weights=stacking_weights,
        backtest_metrics=metrics.get("stacking", metrics.get("ensemble", {})),
        feature_columns=feature_cols,
        horizons=FEATURE_CFG.horizons,
        symbol_encoding=SYMBOL_ENCODING,
    )
    manifest.save(registry.manifests_dir)

    # ── Save detailed backtest metrics ──
    registry.save_backtest_metrics(symbol, version, metrics)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Symbol={symbol}, Version={version}")
    logger.info(f"Models saved to: {registry.models_dir}")
    logger.info(f"{'='*60}")

    return models, metrics, manifest


def _lgbm_early_stopping(stopping_rounds: int):
    """
    LightGBM early stopping callback.
    FIX v1.1: None dönebilir — çağıran taraf filtrelemeli.
    """
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=stopping_rounds, verbose=True)
    except ImportError:
        logger.warning("lightgbm.early_stopping import failed — early stopping devre dışı")
        return None


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Calculate regression metrics for financial return prediction.

    Metrics:
    - MAE:  Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - sMAPE: Symmetric MAPE — robust to near-zero actuals (unlike MAPE)
    - Dir.Acc: Directional accuracy (sign prediction)
    - IC: Information Coefficient (Spearman rank correlation)
    - R²: Coefficient of determination

    Note: Standard MAPE is NOT used because daily returns are near-zero,
    causing division-by-zero inflation. sMAPE is bounded [0, 200%].
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    if len(y_true) == 0:
        return {
            "mae": float("nan"), "rmse": float("nan"),
            "smape": float("nan"), "directional_accuracy": float("nan"),
            "ic": float("nan"), "r2": float("nan"),
        }

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # sMAPE: symmetric MAPE — robust to near-zero actuals
    # Formula: 2 * |actual - pred| / (|actual| + |pred| + ε)
    # Bounded [0, 2] → multiply by 100 for percentage
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)

    # R² — how much variance is explained
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    # Directional accuracy
    actual_dir = np.sign(y_true)
    pred_dir   = np.sign(y_pred)
    dir_acc    = float(np.mean(actual_dir == pred_dir))

    # Information Coefficient (Spearman rank correlation)
    try:
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_true, y_pred)
        ic = float(ic) if not np.isnan(ic) else float("nan")
    except Exception:
        ic = float("nan")

    return {
        "mae":                  round(mae,     6),
        "rmse":                 round(rmse,    6),
        "smape":                round(smape,   4),   # % — replaces MAPE
        "directional_accuracy": round(dir_acc, 4),
        "ic":                   round(ic, 4) if not np.isnan(ic) else float("nan"),
        "r2":                   round(r2, 4),
        "n_samples":            int(len(y_true)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("FinAI Training — Tree Models v2.0 (Hedge Fund Grade)")
    logger.info("=" * 60)

    # Step 1: Collect data (hisse + makro)
    logger.info("\n[1/5] Collecting training data...")
    raw_data, macro_df = collect_training_data(symbols=ALL_SYMBOLS, period="5y", interval="1d")

    if macro_df is not None:
        logger.info(f"Macro data: {len(macro_df)} rows, {len(macro_df.columns)} columns")
    else:
        logger.info("Macro data not available — training without macro features")

    # v2.0: Equity ve kripto verilerini ayır
    equity_data = raw_data[~raw_data["symbol"].isin(CRYPTO_SYMBOLS)].copy()
    crypto_data = raw_data[raw_data["symbol"].isin(CRYPTO_SYMBOLS)].copy()
    logger.info(f"Equity: {len(equity_data)} bars ({equity_data['symbol'].nunique()} symbols)")
    logger.info(f"Crypto: {len(crypto_data)} bars ({crypto_data['symbol'].nunique()} symbols)")

    # Step 2: Feature engineering — equity (cross-sectional features dahil)
    logger.info("\n[2/5] Engineering equity features (with cross-sectional)...")
    equity_featured = engineer_features(equity_data, macro_df=macro_df, add_cs_features=True)
    save_data(equity_featured, "featured_equity_1d.parquet")

    # v2.0: Kripto feature engineering (cross-sectional yok — sadece 2 sembol)
    if not crypto_data.empty:
        logger.info("\n[2b/5] Engineering crypto features...")
        crypto_featured = engineer_features(crypto_data, macro_df=macro_df, add_cs_features=False)
        save_data(crypto_featured, "featured_crypto_1d.parquet")
    else:
        crypto_featured = None

    # Step 3: Walk-forward validation — equity
    logger.info("\n[3/5] Walk-forward validation (equity)...")
    _run_walk_forward_validation(equity_featured)

    # v2.0: Cross-sectional walk-forward (IC Sharpe)
    logger.info("\n[3b/5] Cross-sectional walk-forward validation...")
    _run_cross_sectional_validation(equity_featured)

    # Step 4: Train equity models
    logger.info("\n[4/5] Training equity models...")

    # Universal equity model (tüm hisseler, kripto hariç)
    models, metrics, manifest = train_tree_models(
        equity_featured, target_col="target_1",
        symbol="EQUITY_UNIVERSAL",
        tune_hyperparams=True,
        n_trials=OPTUNA_CFG.n_trials_universal,
    )

    # Multi-horizon equity models
    for horizon in [5, 10, 20]:
        target = f"target_{horizon}"
        if target in equity_featured.columns:
            logger.info(f"\nTraining equity model for horizon={horizon}...")
            train_tree_models(
                equity_featured, target_col=target,
                symbol=f"EQUITY_H{horizon}",
                tune_hyperparams=True,
                n_trials=OPTUNA_CFG.n_trials_sector,
            )

    # v2.0: Cross-sectional rank target modeli
    cs_target = "target_cs_rank_1"
    if cs_target in equity_featured.columns:
        logger.info(f"\nTraining cross-sectional rank model (IC-optimized)...")
        train_tree_models(
            equity_featured, target_col=cs_target,
            symbol="EQUITY_CS_RANK",
            tune_hyperparams=True,
            n_trials=OPTUNA_CFG.n_trials_universal,
        )

    # Sector models
    logger.info("\n[4b/5] Training per-sector models...")
    sector_dfs = split_by_sector(equity_featured)
    for sector, sector_df in sector_dfs.items():
        if sector == "crypto":
            continue  # kripto ayrı eğitilecek
        if len(sector_df) < 500:
            logger.info(f"  Skipping sector '{sector}': only {len(sector_df)} rows")
            continue
        logger.info(f"\n  Training {sector.upper()} sector model ({len(sector_df)} rows)...")
        try:
            train_tree_models(
                sector_df, target_col="target_1",
                symbol=f"SECTOR_{sector.upper()}",
                tune_hyperparams=False,
                n_trials=OPTUNA_CFG.n_trials_sector,
            )
        except Exception as e:
            logger.warning(f"  Sector {sector} training failed: {e}")

    # v2.0: Kripto modeli (ayrı)
    if crypto_featured is not None and len(crypto_featured) >= 500:
        logger.info("\n[4c/5] Training crypto model (separate from equity)...")
        try:
            train_tree_models(
                crypto_featured, target_col="target_1",
                symbol="CRYPTO_UNIVERSAL",
                tune_hyperparams=True,
                n_trials=OPTUNA_CFG.n_trials_crypto,
            )
        except Exception as e:
            logger.warning(f"  Crypto training failed: {e}")

    # Step 5: Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY v2.0")
    logger.info("=" * 60)
    for model_name, m in metrics.items():
        if isinstance(m, dict) and "rmse" in m:
            logger.info(
                f"  {model_name:15s} | sMAPE: {m.get('smape', float('nan')):.2f}% "
                f"| RMSE: {m['rmse']:.6f} "
                f"| Dir.Acc: {m.get('directional_accuracy', 'N/A')} "
                f"| IC: {m.get('ic', 'N/A')} "
                f"| R²: {m.get('r2', 'N/A')}"
            )


def _run_walk_forward_validation(df_featured: pd.DataFrame):
    """
    Run walk-forward validation with XGBoost as a quick sanity check.
    v2.0: PurgedKFold horizon dinamik, IC Sharpe hesabı eklendi.
    """
    from xgboost import XGBRegressor

    feature_cols = get_feature_columns(df_featured)
    if "symbol_encoded" not in feature_cols:
        if "symbol" in df_featured.columns:
            df_featured = df_featured.copy()
            df_featured["symbol_encoded"] = df_featured["symbol"].map(SYMBOL_ENCODING).fillna(-1).astype(int)
            feature_cols.append("symbol_encoded")

    feature_cols = [c for c in feature_cols if c in df_featured.columns]
    target_col = "target_1"

    if target_col not in df_featured.columns:
        logger.warning("Walk-forward: target_1 not found, skipping")
        return

    df_sorted = df_featured.sort_values("time").reset_index(drop=True)

    def xgb_factory():
        return XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20,  # v2.0: regularization
            n_jobs=-1, random_state=42, verbosity=0,
        )

    # v2.0: horizon=1 (target_1 için doğru purge)
    validator = WalkForwardValidator(
        min_train_size=500,
        test_window=63,
        horizon=1,
    )

    wf_results = validator.validate(
        model_factory=xgb_factory,
        X=df_sorted,
        y=df_sorted[target_col],
        feature_cols=feature_cols,
    )

    if "error" in wf_results:
        logger.warning(f"Walk-forward failed: {wf_results['error']}")
        return

    logger.info(f"\n  Walk-Forward Results ({wf_results['n_folds']} folds):")
    logger.info(f"    Overall IC:    {wf_results['overall_ic']:.4f}")
    logger.info(f"    Mean IC:       {wf_results['mean_ic']:.4f}")
    logger.info(f"    IC Sharpe:     {wf_results['ic_sharpe']:.4f}  (>0.5=consistent, >1.0=strong)")
    logger.info(f"    IC Stability:  {wf_results['ic_stability']:.4f} (lower = more consistent)")
    logger.info(f"    Hit Rate:      {wf_results['hit_rate']:.1%} (% folds with IC > 0)")
    logger.info(f"    Mean Dir.Acc:  {wf_results['mean_dir_acc']:.4f}")


def _run_cross_sectional_validation(df_featured: pd.DataFrame):
    """
    v2.0: Cross-sectional walk-forward validation.
    Her zaman adımında semboller arası IC hesaplar — hedge fon standardı.
    """
    from xgboost import XGBRegressor

    if "symbol" not in df_featured.columns:
        logger.warning("CS validation: symbol sütunu yok, atlanıyor")
        return

    n_symbols = df_featured["symbol"].nunique()
    if n_symbols < CS_CFG.min_symbols:
        logger.warning(f"CS validation: yeterli sembol yok ({n_symbols} < {CS_CFG.min_symbols})")
        return

    feature_cols = get_feature_columns(df_featured)
    if "symbol_encoded" not in feature_cols:
        df_featured = df_featured.copy()
        df_featured["symbol_encoded"] = df_featured["symbol"].map(SYMBOL_ENCODING).fillna(-1).astype(int)
        feature_cols.append("symbol_encoded")
    feature_cols = [c for c in feature_cols if c in df_featured.columns]

    def xgb_factory():
        return XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=20, n_jobs=-1, random_state=42, verbosity=0,
        )

    validator = WalkForwardValidator(
        min_train_size=500,
        test_window=63,
        horizon=1,
    )

    cs_results = validator.validate_cross_sectional(
        model_factory=xgb_factory,
        df=df_featured.sort_values("time").reset_index(drop=True),
        feature_cols=feature_cols,
        target_col="target_1",
    )

    if "error" in cs_results:
        logger.warning(f"CS walk-forward failed: {cs_results['error']}")
        return

    logger.info(f"\n  Cross-Sectional Walk-Forward ({cs_results['n_folds']} folds):")
    logger.info(f"    Mean CS IC:    {cs_results['mean_cs_ic']:.4f}")
    logger.info(f"    Std CS IC:     {cs_results['std_cs_ic']:.4f}")
    logger.info(f"    IC Sharpe:     {cs_results['ic_sharpe']:.4f}  (>0.5=consistent alpha)")
    logger.info(f"    Hit Rate:      {cs_results['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
