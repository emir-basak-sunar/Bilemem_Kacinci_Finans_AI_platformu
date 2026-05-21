@echo off
echo ============================================================
echo  FinAI MLflow Tracking Server
echo ============================================================
echo.
echo MLflow UI: http://localhost:5000
echo DB: sqlite:///mlflow.db
echo.

cd /d "%~dp0"

REM MLflow UI'yi başlat
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

pause
