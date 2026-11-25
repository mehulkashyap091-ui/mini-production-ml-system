import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier

from features import build_feature_store, retrieve_candidates, extract_features_for_user, FEATURE_COLUMNS

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
ARTIFACTS_DIR = "artifacts"

INTERNAL_CUTOFF = pd.Timestamp("2011-08-15")
PRODUCTION_CUTOFF = pd.Timestamp("2011-10-01")
N_CANDIDATES = 50

def build_ranking_dataset(store: dict, label_df: pd.DataFrame, n_candidates: int = N_CANDIDATES):
    user_targets = label_df.groupby("CustomerID")["StockCode"].apply(set).to_dict()
    X_parts, y_parts = [], []

    for user_id, true_items in user_targets.items():
        if user_id not in store["user_features"]:
            continue  # Train ranker on warm users; cold-start handled by fallback
        candidates, _ = retrieve_candidates(user_id, store, n_candidates=n_candidates)
        feats_df = extract_features_for_user(user_id, candidates, store)
        labels = [1 if item_id in true_items else 0 for item_id, _ in candidates]
        X_parts.append(feats_df)
        y_parts.extend(labels)

    X = pd.concat(X_parts, ignore_index=True)
    y = np.array(y_parts, dtype=np.int32)
    return X, y

def train():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    # 1. Internal leak-free split inside train_df for Stage 2 Ranker training
    hist_df = train_df[train_df["InvoiceDate"] < INTERNAL_CUTOFF].copy()
    label_df = train_df[train_df["InvoiceDate"] >= INTERNAL_CUTOFF].copy()

    print("[INFO] Building internal training feature store (< 2011-08-15)...")
    hist_store = build_feature_store(hist_df, INTERNAL_CUTOFF)

    print("[INFO] Generating candidate pairs & labels for ranker training...")
    X_train, y_train = build_ranking_dataset(hist_store, label_df, N_CANDIDATES)

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"[INFO] Ranker training set: {len(y_train):,} pairs | Positives: {pos_count:,} ({pos_count/len(y_train):.2%}) | scale_pos_weight: {scale_pos_weight:.2f}")

    # 2. Fit preprocessing + class-weighted LightGBM inside a single Scikit-Learn Pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])

    print("[INFO] Fitting Stage 2 LightGBM Ranker pipeline...")
    pipeline.fit(X_train, y_train)

    # 3. Build production feature store strictly on full train_df (< 2011-10-01)
    print("[INFO] Building production feature store strictly on Train (< 2011-10-01)...")
    prod_store = build_feature_store(train_df, PRODUCTION_CUTOFF)

    # Sanity-check pairwise PR-AUC on Validation set
    X_val, y_val = build_ranking_dataset(prod_store, val_df, N_CANDIDATES)
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    print(f"[INFO] Validation Pairwise PR-AUC: {val_pr_auc:.4f}")

    # 4. Serialize artifacts cleanly with joblib
    prod_store["metadata"] = {
        "model_version": "1.0.0",
        "train_pairs": len(y_train),
        "train_positive_rate": round(pos_count / len(y_train), 4),
        "scale_pos_weight": round(scale_pos_weight, 2),
        "val_pairwise_pr_auc": round(val_pr_auc, 4),
        "n_candidates_stage1": N_CANDIDATES,
    }

    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "ranker_pipeline.joblib"))
    joblib.dump(prod_store, os.path.join(ARTIFACTS_DIR, "feature_store.joblib"))

    # Save feature importances for the report
    importances = dict(zip(FEATURE_COLUMNS, [int(x) for x in pipeline.named_steps["model"].feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    with open(os.path.join(ARTIFACTS_DIR, "training_summary.json"), "w") as f:
        json.dump({"metadata": prod_store["metadata"], "feature_importances": importances}, f, indent=2)

    print("[SUCCESS] Exported artifacts/ranker_pipeline.joblib and artifacts/feature_store.joblib")
    print("Top Feature Importances:", json.dumps(importances, indent=2))

if __name__ == "__main__":
    train()
