"""
============================================================
FinAI Training — Colab Çalıştırma Scripti
============================================================
Bu dosyayı Colab'da her hücreyi ayrı ayrı çalıştır.
Her "# %%"  işareti yeni bir Colab hücresi demek.

ADIMLAR:
1. training/ klasörünü zipleyip Colab'a yükle
2. Bu dosyayı aç, hücreleri sırayla çalıştır
============================================================
"""

# %% ===================== HÜCRE 1: TEMİZLİK =====================
# Eski dosyaları temizle (varsa)
import shutil, os
if os.path.exists('/content/training'):
    shutil.rmtree('/content/training')
    print("Eski training klasörü silindi")
else:
    print("Temiz başlangıç")

# %% ===================== HÜCRE 2: ZIP YÜKLE VE AÇ =====================
# Sol panelden training.zip dosyasını /content/ dizinine yükle
# Sonra bu hücreyi çalıştır:

import zipfile, os

zip_path = '/content/training.zip'
if not os.path.exists(zip_path):
    # Google Drive'dan da yüklenebilir:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # zip_path = '/content/drive/MyDrive/training.zip'
    
    # Veya dosya yükleme dialogu:
    from google.colab import files
    uploaded = files.upload()
    zip_path = '/content/training.zip'

# ZIP'i aç
with zipfile.ZipFile(zip_path, 'r') as z:
    # ZIP içindeki yapıyı kontrol et
    first_entry = z.namelist()[0]
    if first_entry.startswith('training/'):
        # training/ klasörü ZIP içinde → /content/'e aç
        z.extractall('/content/')
        print("ZIP açıldı → /content/training/")
    else:
        # Dosyalar doğrudan ZIP kökünde → training/ klasörü oluştur
        z.extractall('/content/training/')
        print("ZIP açıldı → /content/training/")

# Doğrulama
assert os.path.exists('/content/training/config.py'), \
    "HATA: /content/training/config.py bulunamadı! ZIP yapısını kontrol et."
assert os.path.exists('/content/training/run_all.py'), \
    "HATA: /content/training/run_all.py bulunamadı!"

print("✅ Dosyalar doğru konumda!")
print("Dosya listesi:")
for f in sorted(os.listdir('/content/training/')):
    print(f"  {'📁' if os.path.isdir(f'/content/training/{f}') else '📄'} {f}")

# %% ===================== HÜCRE 3: BAĞIMLILIKLAR =====================
# ⚠️ torch YÜKLEME — Colab'da zaten var, yeniden yüklemek CUDA'yı bozar!

!pip install -q ta pmdarima arch optuna shap yfinance \
    catboost lightgbm xgboost \
    onnx onnxruntime skl2onnx pyarrow \
    joblib tqdm python-dotenv

print("✅ Bağımlılıklar yüklendi!")

# %% ===================== HÜCRE 4: GPU KONTROL =====================
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Memory:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️ GPU yok — DL modelleri CPU'da çalışacak (yavaş)")

# %% ===================== HÜCRE 5: PATH AYARLA =====================
import sys
if '/content' not in sys.path:
    sys.path.insert(0, '/content')

# Import testi
try:
    from training.config import CORE_SYMBOLS, ALL_SYMBOLS, TRAINING_DIR
    print(f"✅ Import başarılı!")
    print(f"   TRAINING_DIR: {TRAINING_DIR}")
    print(f"   Semboller: {len(ALL_SYMBOLS)} ({len(CORE_SYMBOLS)} core)")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("   /content/training/ dizinini kontrol et!")

# %% ===================== HÜCRE 6A: SADECE TREE MODELLERİ (HIZLI) =====================
# İlk test için bunu çalıştır — ~15-30 dakika sürer
# XGBoost, LightGBM, CatBoost eğitir

!cd /content && python -m training.run_all --only-tree

# %% ===================== HÜCRE 6B: SADECE STATISTICAL (CPU) =====================
# ARIMA, SARIMA, SARIMAX, GARCH — ~5-10 dakika

# !cd /content && python -m training.run_all --only-statistical

# %% ===================== HÜCRE 6C: SADECE DL (GPU GEREKLİ) =====================
# BiLSTM, TCN, TFT — ~30-60 dakika (A100'de)

# !cd /content && python -m training.run_all --only-dl

# %% ===================== HÜCRE 6D: TÜM PIPELINE =====================
# Statistical → Tree → DL — ~1-2 saat

# !cd /content && python -m training.run_all

# %% ===================== HÜCRE 7: SONUÇLARI KONTROL ET =====================
import json, os

registry_dir = '/content/training/model_registry'

# Manifest'leri listele
manifests = sorted([f for f in os.listdir(os.path.join(registry_dir, 'manifests')) if f.endswith('.json')])
print(f"📋 {len(manifests)} manifest bulundu:")
for m in manifests:
    print(f"   {m}")

# Metrikleri göster (sadece backtest dosyaları — SHAP dosyaları list formatında)
metrics_dir = os.path.join(registry_dir, 'metrics')
if os.path.exists(metrics_dir):
    for f in sorted(os.listdir(metrics_dir)):
        if f.endswith('_backtest.json'):
            path = os.path.join(metrics_dir, f)
            with open(path) as fp:
                data = json.load(fp)
            print(f"\n📊 {f}:")
            if isinstance(data, dict):
                for model, m in data.items():
                    if isinstance(m, dict) and 'rmse' in m:
                        ic = m.get('ic', 'N/A')
                        rmse = m.get('rmse', 'N/A')
                        da = m.get('directional_accuracy', 'N/A')
                        sharpe = m.get('quant_metrics', {}).get('sharpe', 'N/A') if isinstance(m.get('quant_metrics'), dict) else 'N/A'
                        print(f"   {model:15s} | RMSE: {rmse} | IC: {ic} | Dir.Acc: {da} | Sharpe: {sharpe}")

# Model dosyalarını listele
models_dir = os.path.join(registry_dir, 'models')
print(f"\n📁 Kaydedilen modeller:")
for model_type in os.listdir(models_dir):
    type_dir = os.path.join(models_dir, model_type)
    if os.path.isdir(type_dir):
        files = os.listdir(type_dir)
        if files:
            total_mb = sum(os.path.getsize(os.path.join(type_dir, f)) / 1e6 for f in files)
            print(f"   {model_type:12s}: {len(files)} dosya ({total_mb:.1f} MB)")

# %% ===================== HÜCRE 8: MODELLERİ İNDİR =====================
# Eğitilen modelleri ZIP olarak indir

import shutil
output_zip = '/content/trained_models'
shutil.make_archive(output_zip, 'zip', registry_dir)
print(f"✅ Model arşivi oluşturuldu: {output_zip}.zip")

# İndir
from google.colab import files
files.download(f'{output_zip}.zip')
