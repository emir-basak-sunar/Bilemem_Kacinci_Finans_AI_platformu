"""
Trained models ZIP'ini model_registry'ye extract eden script.
Çalıştır: python extract_models.py
"""
import zipfile
import os
import shutil

ZIP_PATH = os.path.join(os.path.dirname(__file__), "trained_models.zip")
TARGET_DIR = os.path.join(os.path.dirname(__file__), "training", "model_registry")

def main():
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: {ZIP_PATH} bulunamadı!")
        return

    # Backup old manifests
    manifests_dir = os.path.join(TARGET_DIR, "manifests")
    if os.path.exists(manifests_dir):
        backup_dir = os.path.join(TARGET_DIR, "manifests_backup")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(manifests_dir, backup_dir)
        print(f"Backed up old manifests to {backup_dir}")

    # Extract ZIP
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        entries = z.namelist()
        print(f"ZIP contains {len(entries)} entries")
        
        extracted = 0
        for entry in entries:
            # Skip directories
            if entry.endswith('/'):
                continue
            
            target_path = os.path.join(TARGET_DIR, entry)
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            
            with z.open(entry) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            extracted += 1
        
        print(f"Extracted {extracted} files to {TARGET_DIR}")

    # List what we got
    for root, dirs, files in os.walk(TARGET_DIR):
        level = root.replace(TARGET_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            size = os.path.getsize(os.path.join(root, file))
            print(f'{subindent}{file} ({size/1024/1024:.1f} MB)' if size > 100000 else f'{subindent}{file} ({size/1024:.1f} KB)')

if __name__ == "__main__":
    main()
