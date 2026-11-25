import os
import json
import pandas as pd

RAW_PATH = "data/Online_Retail.xlsx"
PROCESSED_DIR = "data/processed"

# Strict temporal cutoffs to prevent look-ahead leakage
VAL_START = pd.Timestamp("2011-10-01")
TEST_START = pd.Timestamp("2011-11-01")

def load_and_clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    print(f"[INFO] Loading raw data from {raw_path} (takes ~20-30s for xlsx)...")
    df = pd.read_excel(raw_path, dtype={"InvoiceNo": str, "StockCode": str})
    initial_rows = len(df)

    # 1. Drop records without CustomerID or StockCode
    df = df.dropna(subset=["CustomerID", "StockCode", "InvoiceDate"]).copy()

    # 2. Remove cancelled orders (InvoiceNo starts with 'C') and non-positive values
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 3. Standardize IDs and types
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["StockCode"] = df["StockCode"].str.strip().str.upper()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # 4. Keep clean product descriptions (fill missing with Unknown)
    df["Description"] = df["Description"].fillna("UNKNOWN").str.strip()

    # 5. Sort chronologically
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    print(f"[INFO] Cleaned rows: {len(df):,} / {initial_rows:,} retained.")
    return df

def temporal_split(df: pd.DataFrame):
    train = df[df["InvoiceDate"] < VAL_START].copy()
    val = df[(df["InvoiceDate"] >= VAL_START) & (df["InvoiceDate"] < TEST_START)].copy()
    test = df[df["InvoiceDate"] >= TEST_START].copy()

    # Explicit Leakage Audit Assertions
    assert train["InvoiceDate"].max() < val["InvoiceDate"].min(), "Leakage: Train overlaps Val!"
    assert val["InvoiceDate"].max() < test["InvoiceDate"].min(), "Leakage: Val overlaps Test!"

    train_users = set(train["CustomerID"])
    val_users = set(val["CustomerID"])
    test_users = set(test["CustomerID"])

    summary = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_date_range": [str(train["InvoiceDate"].min()), str(train["InvoiceDate"].max())],
        "val_date_range": [str(val["InvoiceDate"].min()), str(val["InvoiceDate"].max())],
        "test_date_range": [str(test["InvoiceDate"].min()), str(test["InvoiceDate"].max())],
        "train_unique_users": len(train_users),
        "train_unique_items": int(train["StockCode"].nunique()),
        "val_warm_users": len(val_users & train_users),
        "val_cold_start_users": len(val_users - train_users),
        "test_warm_users": len(test_users & train_users),
        "test_cold_start_users": len(test_users - train_users),
    }
    return train, val, test, summary

def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_and_clean()
    train, val, test, summary = temporal_split(df)

    train.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    val.to_parquet(os.path.join(PROCESSED_DIR, "val.parquet"), index=False)
    test.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"), index=False)

    with open(os.path.join(PROCESSED_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LEAK-FREE TEMPORAL SPLIT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("[SUCCESS] Saved train/val/test parquet files to data/processed/")

if __name__ == "__main__":
    run_pipeline()
