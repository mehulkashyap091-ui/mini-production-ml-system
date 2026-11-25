import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import retrieve_candidates, extract_features_for_user

TEST_PATH = "data/processed/test.parquet"
ARTIFACTS_DIR = "artifacts"

def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def compute_user_metrics(recommended: list, relevant: set, k: int) -> dict:
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return {
        f"Precision@{k}": hits / k,
        f"Recall@{k}": hits / len(relevant) if relevant else 0.0,
        f"NDCG@{k}": ndcg_at_k(top_k, relevant, k),
        f"HitRate@{k}": 1.0 if hits > 0 else 0.0,
    }

def evaluate():
    pipeline = joblib.load(os.path.join(ARTIFACTS_DIR, "ranker_pipeline.joblib"))
    store = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_store.joblib"))
    test_df = pd.read_parquet(TEST_PATH)

    user_ground_truth = test_df.groupby("CustomerID")["StockCode"].apply(set).to_dict()
    pop_top_items = [item for item, _ in store["popular_items"][:20]]

    results = {
        "all_users": {"baseline": [], "two_stage": []},
        "warm_users": {"baseline": [], "two_stage": []},
    }

    for user_id, relevant_items in user_ground_truth.items():
        # Baseline: Global Popularity
        base_m5 = compute_user_metrics(pop_top_items, relevant_items, 5)
        base_m10 = compute_user_metrics(pop_top_items, relevant_items, 10)
        base_combined = {**base_m5, **base_m10}

        # Two-Stage Pipeline: Retrieval -> LightGBM Ranker (with cold-start fallback)
        candidates, is_cold_start = retrieve_candidates(user_id, store, n_candidates=50)
        if is_cold_start:
            ranked_items = pop_top_items
        else:
            X_cand = extract_features_for_user(user_id, candidates, store)
            scores = pipeline.predict_proba(X_cand)[:, 1]
            order = np.argsort(scores)[::-1]
            ranked_items = [candidates[i][0] for i in order]

        rank_m5 = compute_user_metrics(ranked_items, relevant_items, 5)
        rank_m10 = compute_user_metrics(ranked_items, relevant_items, 10)
        rank_combined = {**rank_m5, **rank_m10}

        results["all_users"]["baseline"].append(base_combined)
        results["all_users"]["two_stage"].append(rank_combined)
        if not is_cold_start:
            results["warm_users"]["baseline"].append(base_combined)
            results["warm_users"]["two_stage"].append(rank_combined)

    summary = {}
    for cohort in ["all_users", "warm_users"]:
        base_df = pd.DataFrame(results[cohort]["baseline"]).mean()
        rank_df = pd.DataFrame(results[cohort]["two_stage"]).mean()
        lift_pct = ((rank_df - base_df) / base_df.replace(0, np.nan)) * 100.0
        summary[cohort] = {
            "n_users": len(results[cohort]["baseline"]),
            "baseline": base_df.round(4).to_dict(),
            "two_stage": rank_df.round(4).to_dict(),
            "relative_lift_pct": lift_pct.round(2).to_dict(),
        }

    with open(os.path.join(ARTIFACTS_DIR, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TEST SET EVALUATION (WARM USERS: n={}) ===".format(summary["warm_users"]["n_users"]))
    warm_tbl = pd.DataFrame({
        "Popularity Baseline": summary["warm_users"]["baseline"],
        "Two-Stage Ranker": summary["warm_users"]["two_stage"],
        "Relative Lift (%)": summary["warm_users"]["relative_lift_pct"],
    })
    print(warm_tbl.to_string())

    print("\n=== TEST SET EVALUATION (ALL USERS incl. Cold-Start: n={}) ===".format(summary["all_users"]["n_users"]))
    all_tbl = pd.DataFrame({
        "Popularity Baseline": summary["all_users"]["baseline"],
        "Two-Stage Ranker": summary["all_users"]["two_stage"],
        "Relative Lift (%)": summary["all_users"]["relative_lift_pct"],
    })
    print(all_tbl.to_string())

    # Generate report-ready bar chart for K=10 metrics (Warm Users)
    metrics_k10 = ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]
    base_vals = [summary["warm_users"]["baseline"][m] for m in metrics_k10]
    rank_vals = [summary["warm_users"]["two_stage"][m] for m in metrics_k10]

    x = np.arange(len(metrics_k10))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    b1 = ax.bar(x - width / 2, base_vals, width, label="Popularity Baseline", color="#94a3b8")
    b2 = ax.bar(x + width / 2, rank_vals, width, label="Two-Stage LightGBM Ranker", color="#2563eb")

    ax.set_ylabel("Score")
    ax.set_title("Test Set Offline Ranking Performance (Warm Users, K=10)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_k10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "eval_comparison.png"))
    print("\n[SUCCESS] Saved artifacts/eval_results.json and artifacts/eval_comparison.png")

if __name__ == "__main__":
    evaluate()
