"""
FinAI Training Orchestrator — run_all.py
==========================================
Tüm modelleri sırayla çalıştırır.

Colab A100'de kullanım:
    python -m training.run_all

Seçici çalıştırma:
    python -m training.run_all --skip-statistical
    python -m training.run_all --skip-dl
    python -m training.run_all --only-tree
"""
import sys
import os
import argparse
import logging
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_statistical():
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Statistical Models (ARIMA/SARIMA/SARIMAX + GARCH)")
    logger.info("=" * 60)
    t0 = time.time()

    from training.models.statistical.train_arima import main as arima_main
    arima_main()

    try:
        from training.models.statistical.train_garch import main as garch_main
        garch_main()
    except Exception as e:
        logger.warning(f"GARCH skipped: {e}")

    logger.info(f"Statistical models done in {(time.time()-t0)/60:.1f} min")


def run_tree():
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Tree Models (XGBoost / LightGBM / CatBoost)")
    logger.info("=" * 60)
    t0 = time.time()

    from training.models.tree.train_tree_models import main as tree_main
    tree_main()

    logger.info(f"Tree models done in {(time.time()-t0)/60:.1f} min")


def run_dl():
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Deep Learning (BiLSTM + TCN + TFT)")
    logger.info("=" * 60)
    t0 = time.time()

    from training.models.dl.train_lstm import main as lstm_main
    lstm_main()

    from training.models.dl.train_tcn_tft import main as tcn_tft_main
    tcn_tft_main()

    logger.info(f"DL models done in {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description="FinAI Training Orchestrator")
    parser.add_argument("--skip-statistical", action="store_true", help="Skip ARIMA/SARIMA/GARCH")
    parser.add_argument("--skip-tree",        action="store_true", help="Skip XGBoost/LightGBM/CatBoost")
    parser.add_argument("--skip-dl",          action="store_true", help="Skip LSTM/TCN/TFT")
    parser.add_argument("--only-tree",        action="store_true", help="Run only tree models")
    parser.add_argument("--only-dl",          action="store_true", help="Run only DL models")
    parser.add_argument("--only-statistical", action="store_true", help="Run only statistical models")
    args = parser.parse_args()

    total_start = time.time()
    logger.info("=" * 60)
    logger.info("FinAI Training Orchestrator v2.0")
    logger.info("=" * 60)

    if args.only_statistical:
        run_statistical()
    elif args.only_tree:
        run_tree()
    elif args.only_dl:
        run_dl()
    else:
        if not args.skip_statistical:
            run_statistical()
        if not args.skip_tree:
            run_tree()
        if not args.skip_dl:
            run_dl()

    total_min = (time.time() - total_start) / 60
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL TRAINING COMPLETE in {total_min:.1f} min")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
