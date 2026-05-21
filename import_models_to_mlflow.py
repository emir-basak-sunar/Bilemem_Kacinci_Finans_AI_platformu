"""
FinAI — trained_models/ → MLflow Import Script
================================================
Bu scripti bir kez çalıştır, tüm eğitilmiş modeller MLflow'a yüklenir.

Kullanım:
    cd C:\\Users\\Emir Başak Sunar\\Desktop\\Bilemem_Kacinci_Finans_AI_platformu
    python import_models_to_mlflow.py

Sonra MLflow UI'yi başlat:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    → http://localhost:5000
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Workspace root'u path'e ekle
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR    = os.path.join(WORKSPACE_ROOT, "backend")

if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

def main():
    logger.info("=" * 60)
    logger.info("FinAI MLflow Import — trained_models/ → MLflow")
    logger.info("=" * 60)

    # MLflow kurulu mu?
    try:
        import mlflow
        logger.info(f"MLflow version: {mlflow.__version__}")
    except ImportError:
        logger.error("MLflow kurulu değil! Kurmak için:")
        logger.error("  pip install mlflow")
        sys.exit(1)

    # trained_models/ var mı?
    trained_models_dir = os.path.join(WORKSPACE_ROOT, "trained_models")
    if not os.path.isdir(trained_models_dir):
        logger.error(f"trained_models/ klasörü bulunamadı: {trained_models_dir}")
        sys.exit(1)

    manifests_dir = os.path.join(trained_models_dir, "manifests")
    import glob
    manifest_files = glob.glob(os.path.join(manifests_dir, "*_v*.json"))
    logger.info(f"Bulunan manifest sayısı: {len(manifest_files)}")

    if not manifest_files:
        logger.error("Hiç manifest dosyası bulunamadı!")
        sys.exit(1)

    # Import
    from backend.services.mlflow_service import import_all_models_to_mlflow, MLFLOW_TRACKING_URI
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    result = import_all_models_to_mlflow()

    logger.info("=" * 60)
    logger.info(f"Import tamamlandı!")
    logger.info(f"  Yeni import: {result.get('imported', 0)}")
    logger.info(f"  Atlandı:     {result.get('skipped', 0)}")
    logger.info(f"  Hata:        {len(result.get('errors', []))}")
    if result.get('errors'):
        for err in result['errors']:
            logger.warning(f"  Hata: {err}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("MLflow UI'yi başlatmak için:")
    logger.info("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    logger.info("  → http://localhost:5000")
    logger.info("")
    logger.info("Veya start_mlflow.bat dosyasını çalıştır.")


if __name__ == "__main__":
    main()
