import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

TRAIN_PATH = "data/processed/train.parquet"
TEST_PATH = "data/processed/test.parquet"
OUT_JSON = "artifacts/drift_report.json"

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Computes Population Stability Index using quantile bins fitted on reference data."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = np.clip(exp_counts / len(expected), 1e-6, None)
    act_pct = np.clip(act_counts / len(actual), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

def classify_psi(psi_val: float) -> str:
    if psi_val < 0.10:
        return "NO_DRIFT"
    elif psi_val < 0.25:
        return "MODERATE_DRIFT_MONITOR"
    return "SIGNIFICANT_DRIFT_RETRAIN"

def run_drift_check():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Order-level basket size (items per invoice)
    train_basket = train_df.groupby("InvoiceNo")["StockCode"].nunique().values
    test_basket = test_df.groupby("InvoiceNo")["StockCode"].nunique().values

    feature_pairs = {
        "UnitPrice": (train_df["UnitPrice"].values, test_df["UnitPrice"].values),
        "Quantity": (train_df["Quantity"].values, test_df["Quantity"].values),
        "TotalAmount": (train_df["TotalAmount"].values, test_df["TotalAmount"].values),
        "BasketSize_UniqueItems": (train_basket, test_basket),
    }

    report = {}
    for feat, (ref, cur) in feature_pairs.items():
        psi_val = calculate_psi(ref, cur)
        ks_stat, ks_pval = ks_2samp(ref, cur)
        report[feat] = {
            "train_median": round(float(np.median(ref)), 3),
            "test_median": round(float(np.median(cur)), 3),
            "psi": round(psi_val, 4),
            "psi_status": classify_psi(psi_val),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(f"{ks_pval:.2e}"),
        }

    # Check catalog & user cold-start shift
    train_items = set(train_df["StockCode"])
    test_items = set(test_df["StockCode"])
    unseen_item_ratio = len(test_items - train_items) / max(len(test_items), 1)

    train_users = set(train_df["CustomerID"])
    test_users = set(test_df["CustomerID"])
    cold_start_user_ratio = len(test_users - train_users) / max(len(test_users), 1)

    output = {
        "reference_window": "2010-12 to 2011-09 (Train)",
        "current_window": "2011-11 to 2011-12 (Test)",
        "feature_drift": report,
        "entity_drift": {
            "unseen_catalog_items_pct": round(unseen_item_ratio * 100.0, 2),
            "cold_start_users_pct": round(cold_start_user_ratio * 100.0, 2),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== EMPIRICAL DATA DRIFT REPORT (TRAIN vs TEST) ===")
    print(json.dumps(output, indent=2))
    print(f"[SUCCESS] Saved {OUT_JSON}")

if __name__ == "__main__":
    run_drift_check()
