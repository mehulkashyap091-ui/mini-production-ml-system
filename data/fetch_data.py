import os
import zipfile
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_XLSX = os.path.join(DATA_DIR, "Online_Retail.xlsx")
ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

def download_dataset():
    if os.path.exists(RAW_XLSX):
        print(f"[INFO] Dataset already exists at {RAW_XLSX}")
        return RAW_XLSX

    zip_path = os.path.join(DATA_DIR, "online_retail.zip")
    print("[INFO] Downloading UCI Online Retail dataset...")
    try:
        resp = requests.get(ZIP_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        with zipfile.ZipFile(zip_path, "r") as z:
            xlsx_name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
            with z.open(xlsx_name) as src, open(RAW_XLSX, "wb") as dst:
                dst.write(src.read())
        os.remove(zip_path)
    except Exception as e:
        print(f"[WARN] Zip download failed ({e}). Trying fallback URL...")
        resp = requests.get(FALLBACK_URL, timeout=60)
        resp.raise_for_status()
        with open(RAW_XLSX, "wb") as f:
            f.write(resp.content)

    print(f"[SUCCESS] Saved raw dataset to {RAW_XLSX}")
    return RAW_XLSX

if __name__ == "__main__":
    download_dataset()
