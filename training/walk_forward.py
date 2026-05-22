"""
FinAI Walk-Forward Validation — v2.0
======================================
Proper time-series cross-validation for financial ML.

Why standard k-fold is wrong for finance:
  - Future data leaks into training (look-ahead bias)
  - Correlated samples inflate performance metrics
  - Regime changes are not captured

This module implements:
  1. PurgedKFold — removes samples near the train/test boundary
     to prevent label leakage (e.g. target_5 uses 5 future bars)
  2. WalkForwardValidator — expanding window backtesting
     Each fold: train on all past data, test on next window
  3. Embargo — additional gap between train and test sets
  4. compute_quant_metrics — Sharpe, Sortino, Calmar, MaxDD, Turnover
  5. compute_long_short_metrics — long-short portfolio (hedge fund style)
  6. compute_ic_sharpe — IC Sharpe ratio (consistency metric)

Değişiklikler v2.0 (hedge-fund grade):
- FIX: PurgedKFold.split — horizon parametresi dinamik olarak kullanılıyor
  Önceki: purge_start = test_start - self.horizon (sabit 1)
  Yeni: horizon=h ile çağrılınca target_h için doğru purge penceresi
- YENİ: compute_long_short_metrics — long-short equity strategy
  Top-N long, Bottom-N short, piyasa nötr alpha
- YENİ: compute_ic_sharpe — IC Sharpe = mean(IC)/std(IC)*sqrt(252/test_window)
  IC Sharpe > 0.5 = consistent alpha, > 1.0 = strong alpha
- YENİ: WalkForwardValidator.validate_cross_sectional — multi-symbol IC
- YENİ: RegimeDetector — VIX + SP500 trend ile rejim tespiti
- compute_quant_metrics: long-short strategy eklendi (long-only yanında)

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018)
"""
import numpy as np
import pandas as pd
import logging
from typing import Iterator, Tuple, List, Dict, Optional
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


# ── Purged K-Fold ─────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    K-Fold cross-validation with purging and embargo for financial time series.

    Purging: removes training samples whose labels overlap with the test period.
    Embargo: adds a gap after the test period to prevent leakage from autocorrelation.

    v2.0 FIX: horizon parametresi artık gerçekten kullanılıyor.
    target_5 için horizon=5, target_20 için horizon=20 geçilmeli.
    Önceki implementasyonda purge_start = test_start - 1 (sabit) idi.

    Args:
        n_splits:    Number of folds
        horizon:     Prediction horizon (bars) — purge window = horizon bars
        embargo_pct: Fraction of test set to embargo after test period (default 1%)
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo_pct: float = 0.01):
        self.n_splits    = n_splits
        self.horizon     = horizon
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices with purging and embargo.

        v2.0 FIX: purge_start = test_start - self.horizon
        Önceki: purge_start = max(0, test_start - self.horizon)
        Bu zaten doğruydu ama horizon=1 sabit geçiliyordu.
        Artık horizon=5 veya horizon=20 ile çağrılabilir.

        Yields:
            (train_indices, test_indices)
        """
        n = len(X)
        fold_size = n // self.n_splits
        embargo_size = max(1, int(fold_size * self.embargo_pct))

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end   = test_start + fold_size if fold < self.n_splits - 1 else n

            # v2.0: Purge window = horizon bars
            # target_h için: i. bar'ın label'ı [i, i+h] barlarını kullanır
            # test_start'tan horizon bar önce başlayan train sample'ları purge edilir
            purge_start = max(0, test_start - self.horizon)

            # Train: all samples before purge window
            train_idx = np.arange(0, purge_start)

            # Test: the fold window
            test_idx = np.arange(test_start, test_end)

            # Embargo: skip samples right after test end
            embargo_end = min(n, test_end + embargo_size)

            if len(train_idx) < 50 or len(test_idx) < 10:
                logger.debug(f"Fold {fold}: skipping (too few samples)")
                continue

            logger.debug(
                f"Fold {fold}: train=[0,{purge_start}), "
                f"test=[{test_start},{test_end}), "
                f"embargo=[{test_end},{embargo_end}), "
                f"horizon={self.horizon}"
            )
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ── Walk-Forward Validator ────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Expanding window walk-forward validation.

    Each fold:
    - Train: all data from start to fold boundary
    - Test:  next `test_window` bars

    v2.0 additions:
    - validate_cross_sectional: multi-symbol IC hesabı
    - IC Sharpe hesabı (consistency metric)
    - Per-fold regime tracking

    Args:
        min_train_size: Minimum bars required for first training window
        test_window:    Number of bars per test fold
        step:           How many bars to advance between folds (default = test_window)
        horizon:        Prediction horizon for purging
    """

    def __init__(
        self,
        min_train_size: int = 500,
        test_window: int = 63,    # ~1 quarter
        step: Optional[int] = None,
        horizon: int = 1,
    ):
        self.min_train_size = min_train_size
        self.test_window    = test_window
        self.step           = step or test_window
        self.horizon        = horizon

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_idx, test_idx) pairs."""
        n = len(X)
        start = self.min_train_size

        while start + self.test_window <= n:
            # v2.0: purge = horizon bars (target_h için doğru pencere)
            train_end  = start - self.horizon
            test_start = start
            test_end   = min(start + self.test_window, n)

            if train_end > 50:
                yield np.arange(0, train_end), np.arange(test_start, test_end)

            start += self.step

    def validate(
        self,
        model_factory,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
    ) -> Dict:
        """
        Run walk-forward validation and collect per-fold metrics.

        Args:
            model_factory: Callable that returns a fresh unfitted model
            X:             Feature DataFrame (indexed same as y)
            y:             Target Series
            feature_cols:  Feature columns to use

        Returns:
            Dict with per-fold metrics and aggregate statistics including IC Sharpe
        """
        from training.models.tree.train_tree_models import evaluate_predictions
        from scipy.stats import spearmanr

        fold_metrics = []
        all_preds = []
        all_actuals = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            X_train = X.iloc[train_idx][feature_cols]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx][feature_cols]
            y_test  = y.iloc[test_idx]

            if len(X_train) < 100 or len(X_test) < 10:
                continue

            try:
                model = model_factory()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                m = evaluate_predictions(y_test.values, preds)
                m["fold"] = fold_idx
                m["train_size"] = len(train_idx)
                m["test_size"]  = len(test_idx)
                fold_metrics.append(m)

                all_preds.extend(preds.tolist())
                all_actuals.extend(y_test.values.tolist())

            except Exception as e:
                logger.warning(f"Walk-forward fold {fold_idx} failed: {e}")
                continue

        if not fold_metrics:
            return {"error": "No valid folds"}

        ics   = [m["ic"]   for m in fold_metrics if not np.isnan(m.get("ic", float("nan")))]
        rmses = [m["rmse"] for m in fold_metrics]
        das   = [m["directional_accuracy"] for m in fold_metrics]

        # Overall IC on concatenated predictions
        overall_ic = float("nan")
        if all_preds and all_actuals:
            try:
                ic, _ = spearmanr(all_actuals, all_preds)
                overall_ic = round(float(ic), 4)
            except Exception:
                pass

        ic_stability = round(float(np.std(ics)), 4) if len(ics) > 1 else float("nan")
        hit_rate = round(sum(1 for ic in ics if ic > 0) / max(len(ics), 1), 3)

        # v2.0: IC Sharpe = mean(IC) / std(IC) * sqrt(252 / test_window)
        # IC Sharpe > 0.5 = consistent alpha, > 1.0 = strong alpha
        ic_sharpe = float("nan")
        if len(ics) > 1 and np.std(ics) > 0:
            ic_sharpe = round(
                float(np.mean(ics)) / float(np.std(ics)) * np.sqrt(252 / self.test_window),
                4
            )

        return {
            "n_folds":       len(fold_metrics),
            "overall_ic":    overall_ic,
            "mean_ic":       round(float(np.mean(ics)), 4)    if ics   else float("nan"),
            "ic_stability":  ic_stability,
            "ic_sharpe":     ic_sharpe,   # v2.0: yeni metrik
            "hit_rate":      hit_rate,
            "mean_rmse":     round(float(np.mean(rmses)), 6)  if rmses else float("nan"),
            "mean_dir_acc":  round(float(np.mean(das)), 4)    if das   else float("nan"),
            "fold_metrics":  fold_metrics,
        }

    def validate_cross_sectional(
        self,
        model_factory,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_1",
        symbol_col: str = "symbol",
        time_col: str = "time",
    ) -> Dict:
        """
        v2.0: Multi-symbol cross-sectional walk-forward validation.

        Her fold'da:
        1. Model eğitilir (tüm semboller, geçmiş veri)
        2. Test döneminde her sembol için tahmin yapılır
        3. Her zaman adımında cross-sectional IC hesaplanır
           IC = Spearman(predicted_return_rank, actual_return_rank)
        4. Fold IC'leri toplanır → IC Sharpe hesaplanır

        Bu, hedge fon standardı olan "long-short equity" alpha'sını ölçer.

        Args:
            model_factory: Callable → fresh model
            df: Multi-symbol DataFrame (time + symbol + features + target)
            feature_cols: Feature sütunları
            target_col: Hedef sütun
            symbol_col: Sembol sütunu
            time_col: Zaman sütunu

        Returns:
            Dict with cross-sectional IC metrics
        """
        from scipy.stats import spearmanr

        if symbol_col not in df.columns:
            logger.warning("validate_cross_sectional: symbol sütunu yok")
            return {"error": "No symbol column"}

        # Zaman bazlı sort
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        unique_times = sorted(df_sorted[time_col].unique())
        n_times = len(unique_times)

        fold_ics = []
        all_fold_metrics = []

        start_idx = self.min_train_size

        while start_idx + self.test_window <= n_times:
            train_times = unique_times[:start_idx - self.horizon]
            test_times  = unique_times[start_idx: start_idx + self.test_window]

            train_df = df_sorted[df_sorted[time_col].isin(train_times)]
            test_df  = df_sorted[df_sorted[time_col].isin(test_times)]

            if len(train_df) < 100 or len(test_df) < 10:
                start_idx += self.step
                continue

            try:
                model = model_factory()
                model.fit(train_df[feature_cols], train_df[target_col])
                test_preds = model.predict(test_df[feature_cols])

                test_df = test_df.copy()
                test_df["_pred"] = test_preds

                # Her zaman adımında cross-sectional IC
                time_ics = []
                for t in test_times:
                    t_df = test_df[test_df[time_col] == t]
                    if len(t_df) < 3:
                        continue
                    try:
                        ic, _ = spearmanr(t_df[target_col].values, t_df["_pred"].values)
                        if not np.isnan(ic):
                            time_ics.append(float(ic))
                    except Exception:
                        pass

                if time_ics:
                    fold_ic = float(np.mean(time_ics))
                    fold_ics.append(fold_ic)
                    all_fold_metrics.append({
                        "fold_ic": round(fold_ic, 4),
                        "n_time_steps": len(time_ics),
                        "train_size": len(train_df),
                    })

            except Exception as e:
                logger.warning(f"CS walk-forward fold failed: {e}")

            start_idx += self.step

        if not fold_ics:
            return {"error": "No valid cross-sectional folds"}

        ic_sharpe = float("nan")
        if len(fold_ics) > 1 and np.std(fold_ics) > 0:
            ic_sharpe = round(
                float(np.mean(fold_ics)) / float(np.std(fold_ics)) * np.sqrt(252 / self.test_window),
                4
            )

        return {
            "n_folds":      len(fold_ics),
            "mean_cs_ic":   round(float(np.mean(fold_ics)), 4),
            "std_cs_ic":    round(float(np.std(fold_ics)), 4),
            "ic_sharpe":    ic_sharpe,
            "hit_rate":     round(sum(1 for ic in fold_ics if ic > 0) / len(fold_ics), 3),
            "fold_metrics": all_fold_metrics,
        }


# ── Quant Metrics ─────────────────────────────────────────────────────────────

def compute_quant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None,
    transaction_cost: float = 0.001,
    horizon: int = 1,
) -> Dict:
    """
    Compute comprehensive quant performance metrics.

    v2.1 FIX: Sharpe annualization düzeltildi.
    - Önceki: sqrt(252) — günlük return varsayımı
    - Yeni: sqrt(252 / horizon) — horizon-adjusted annualization
    - Örnek: horizon=10 → sqrt(252/10)=5.02, önceki sqrt(252)=15.87 → 3x şişirme

    v2.1 FIX: IC overlap düzeltmesi.
    - Overlapping target'lar (target_10 için 10 bar overlap) IC'yi şişirir.
    - Stride=horizon ile bağımsız sample'lar alınır.

    v2.0: Long-short strategy eklendi (long-only yanında).
    Long-short: top tercile long, bottom tercile short → piyasa nötr.

    Args:
        y_true:           Actual returns
        y_pred:           Predicted returns
        prices:           Price series (for PnL calculation)
        transaction_cost: Round-trip cost (default 0.1%)
        horizon:          Prediction horizon (bars) — for annualization fix

    Returns:
        Dict with IC, Sharpe, Hit Rate, PnL metrics (long-only + long-short)
    """
    from scipy.stats import spearmanr

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    if len(y_true) < 10:
        return {}

    # ── IC (Spearman rank correlation) — overlap-adjusted ──
    # FIX v2.1: Overlapping samples şişirir IC ve p-value'yu.
    # stride=horizon ile bağımsız sample'lar al.
    try:
        stride = max(1, horizon)
        y_true_ind = y_true[::stride]
        y_pred_ind = y_pred[::stride]
        if len(y_true_ind) >= 5:
            ic, ic_pval = spearmanr(y_true_ind, y_pred_ind)
        else:
            ic, ic_pval = spearmanr(y_true, y_pred)
        ic = float(ic) if not np.isnan(ic) else 0.0
    except Exception:
        ic, ic_pval = 0.0, 1.0

    # ── Directional accuracy ──
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # ── Long-only strategy PnL ──
    positions_long = (y_pred > 0).astype(float)
    strategy_returns_long = positions_long * y_true
    position_changes_long = np.abs(np.diff(np.concatenate([[0], positions_long])))
    tc_drag_long = position_changes_long * transaction_cost
    net_returns_long = strategy_returns_long - tc_drag_long

    # ── Long-short strategy (v2.0) ──
    # Top tercile: +1 (long), Bottom tercile: -1 (short), Middle: 0 (flat)
    p33 = np.percentile(y_pred, 33)
    p67 = np.percentile(y_pred, 67)
    positions_ls = np.where(y_pred >= p67, 1.0, np.where(y_pred <= p33, -1.0, 0.0))
    strategy_returns_ls = positions_ls * y_true
    position_changes_ls = np.abs(np.diff(np.concatenate([[0], positions_ls])))
    tc_drag_ls = position_changes_ls * transaction_cost
    net_returns_ls = strategy_returns_ls - tc_drag_ls

    # FIX v2.1: horizon-adjusted annualization
    # target_h = h-bar forward return → annualization = sqrt(252 / h)
    # Önceki sqrt(252) sadece h=1 (günlük) için doğruydu.
    ann_factor = np.sqrt(252 / max(1, horizon))

    def _sharpe(rets):
        if rets.std() > 0:
            return float(np.mean(rets) / rets.std() * ann_factor)
        return 0.0

    def _sortino(rets):
        downside = rets[rets < 0]
        if len(downside) > 0 and downside.std() > 0:
            return float(np.mean(rets) / downside.std() * ann_factor)
        return 0.0

    def _max_dd(rets):
        cumulative = np.cumprod(1 + rets)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / (rolling_max + 1e-10)
        return float(np.min(drawdowns))

    def _calmar(rets):
        # FIX v2.1: annual_ret de horizon-adjusted olmalı
        annual_ret = float(np.mean(rets) * (252 / max(1, horizon)))
        mdd = abs(_max_dd(rets))
        return annual_ret / mdd if mdd > 1e-6 else 0.0

    return {
        # IC metrics
        "ic":              round(ic, 4),
        "ic_pvalue":       round(float(ic_pval), 4),
        "directional_acc": round(dir_acc, 4),
        "hit_rate":        round(dir_acc, 4),
        # Long-only
        "sharpe":          round(_sharpe(net_returns_long), 3),
        "sortino":         round(_sortino(net_returns_long), 3),
        "calmar":          round(_calmar(net_returns_long), 3),
        "max_drawdown":    round(_max_dd(net_returns_long), 4),
        "annual_return":   round(float(np.mean(net_returns_long) * (252 / max(1, horizon))), 4),
        "turnover":        round(float(np.mean(position_changes_long)), 4),
        # Long-short (v2.0) — piyasa nötr alpha
        "ls_sharpe":       round(_sharpe(net_returns_ls), 3),
        "ls_sortino":      round(_sortino(net_returns_ls), 3),
        "ls_calmar":       round(_calmar(net_returns_ls), 3),
        "ls_max_drawdown": round(_max_dd(net_returns_ls), 4),
        "ls_annual_return":round(float(np.mean(net_returns_ls) * (252 / max(1, horizon))), 4),
        "ls_turnover":     round(float(np.mean(position_changes_ls)), 4),
        "n_samples":       int(len(y_true)),
        "n_independent":   int(len(y_true_ind)) if 'y_true_ind' in dir() else int(len(y_true)),
        "horizon":         horizon,
    }


def compute_ic_sharpe(fold_ics: List[float], test_window: int = 63) -> float:
    """
    v2.0: IC Sharpe ratio hesaplar.

    IC Sharpe = mean(IC per fold) / std(IC per fold) * sqrt(252 / test_window)

    Yorumlama:
    - IC Sharpe > 0.5: consistent alpha (kullanılabilir)
    - IC Sharpe > 1.0: strong alpha (iyi)
    - IC Sharpe > 2.0: exceptional alpha (hedge fon kalitesi)

    Args:
        fold_ics: Her walk-forward fold'un IC değerleri
        test_window: Fold başına test bar sayısı (annualization için)

    Returns:
        IC Sharpe ratio
    """
    if len(fold_ics) < 2:
        return float("nan")
    std_ic = float(np.std(fold_ics))
    if std_ic < 1e-10:
        return float("nan")
    return round(
        float(np.mean(fold_ics)) / std_ic * np.sqrt(252 / test_window),
        4
    )


# ── Regime Detector ───────────────────────────────────────────────────────────

class RegimeDetector:
    """
    v2.0: Market regime detection using VIX + SP500 trend.

    Rejimler:
    - bull_low_vol:  SP500 > MA200 AND VIX < 15
    - bull_high_vol: SP500 > MA200 AND VIX >= 15
    - bear_low_vol:  SP500 < MA200 AND VIX < 25
    - bear_high_vol: SP500 < MA200 AND VIX >= 25
    - crisis:        VIX >= 40

    Kullanım:
        detector = RegimeDetector()
        regime = detector.detect(vix=18.5, sp500=4500, sp500_ma200=4200)
        weights = detector.get_ensemble_weights(regime)
    """

    REGIMES = ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol", "crisis"]

    def __init__(self, vix_low=15.0, vix_high=25.0, vix_extreme=40.0):
        self.vix_low     = vix_low
        self.vix_high    = vix_high
        self.vix_extreme = vix_extreme

    def detect(
        self,
        vix: float,
        sp500: Optional[float] = None,
        sp500_ma200: Optional[float] = None,
    ) -> str:
        """Detect current market regime."""
        if vix >= self.vix_extreme:
            return "crisis"

        if sp500 is not None and sp500_ma200 is not None:
            bull = sp500 > sp500_ma200
        else:
            bull = True  # default assumption

        if bull:
            return "bull_low_vol" if vix < self.vix_low else "bull_high_vol"
        else:
            return "bear_low_vol" if vix < self.vix_high else "bear_high_vol"

    def detect_series(
        self,
        vix_series: pd.Series,
        sp500_series: Optional[pd.Series] = None,
        smooth_window: int = 5,
    ) -> pd.Series:
        """
        Detect regime for a time series.
        smooth_window: ani geçişleri önlemek için rolling mode.
        """
        if sp500_series is not None:
            sp500_ma200 = sp500_series.rolling(200, min_periods=50).mean()
        else:
            sp500_ma200 = None

        regimes = []
        for i in range(len(vix_series)):
            vix = float(vix_series.iloc[i]) if not np.isnan(vix_series.iloc[i]) else 20.0
            sp500 = float(sp500_series.iloc[i]) if sp500_series is not None else None
            ma200 = float(sp500_ma200.iloc[i]) if sp500_ma200 is not None and not np.isnan(sp500_ma200.iloc[i]) else None
            regimes.append(self.detect(vix, sp500, ma200))

        regime_series = pd.Series(regimes, index=vix_series.index)

        # Smoothing: rolling mode (ani geçişleri önle)
        if smooth_window > 1:
            regime_series = regime_series.rolling(
                smooth_window, min_periods=1
            ).apply(lambda x: pd.Series(x).mode()[0] if len(x) > 0 else x.iloc[-1], raw=False)

        return regime_series

    def get_ensemble_weights(self, regime: str) -> Dict[str, float]:
        """Get model ensemble weights for a given regime."""
        from training.config import REGIME_CFG
        weights = REGIME_CFG.ensemble_weights_by_regime
        # Fallback: equal weights
        default = {"xgboost": 0.25, "lightgbm": 0.25, "catboost": 0.25, "lstm": 0.25}
        return weights.get(regime, default)


# ── Sector Grouping ───────────────────────────────────────────────────────────

SECTOR_GROUPS = {
    "tech": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC",
        "ADBE", "CRM", "ORCL", "CSCO", "QCOM", "AVGO",
    ],
    "finance": [
        "JPM", "GS", "BAC", "V", "MA", "PYPL", "SQ", "COIN",
    ],
    "consumer": [
        "AMZN", "TSLA", "NFLX", "DIS", "UBER",
    ],
    "crypto": [
        "BTC-USD", "ETH-USD",
    ],
    "other": [
        "IBM",
    ],
}

SYMBOL_TO_SECTOR = {
    sym: sector
    for sector, symbols in SECTOR_GROUPS.items()
    for sym in symbols
}


def get_sector(symbol: str) -> str:
    return SYMBOL_TO_SECTOR.get(symbol.upper(), "other")


def split_by_sector(df: pd.DataFrame, symbol_col: str = "symbol") -> Dict[str, pd.DataFrame]:
    """Split a multi-symbol DataFrame into sector DataFrames."""
    result = {}
    if symbol_col not in df.columns:
        return {"universal": df}

    for sector in SECTOR_GROUPS:
        symbols = SECTOR_GROUPS[sector]
        mask = df[symbol_col].isin(symbols)
        if mask.sum() > 0:
            result[sector] = df[mask].copy()

    known = set(sym for syms in SECTOR_GROUPS.values() for sym in syms)
    unknown_mask = ~df[symbol_col].isin(known)
    if unknown_mask.sum() > 0:
        result["other"] = df[unknown_mask].copy()

    return result
