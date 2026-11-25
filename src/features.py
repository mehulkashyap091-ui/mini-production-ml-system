import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLUMNS = [
    "retrieval_score",
    "user_bought_before",
    "user_item_prev_qty",
    "user_item_recency_days",
    "user_total_orders",
    "user_total_qty",
    "user_avg_order_spend",
    "user_recency_days",
    "item_total_orders",
    "item_unique_buyers",
    "item_avg_price",
    "item_recency_days",
    "price_ratio_to_user_avg",
]

def build_feature_store(df: pd.DataFrame, cutoff_date: pd.Timestamp, top_neighbors: int = 40) -> dict:
    """Fits all user/item stats and Stage 1 item-item cosine similarity strictly on df."""
    # 1. User-level features
    u_agg = df.groupby("CustomerID").agg(
        user_total_orders=("InvoiceNo", "nunique"),
        user_total_qty=("Quantity", "sum"),
        user_total_spend=("TotalAmount", "sum"),
        user_avg_item_price=("UnitPrice", "mean"),
        last_date=("InvoiceDate", "max"),
    )
    u_agg["user_avg_order_spend"] = u_agg["user_total_spend"] / u_agg["user_total_orders"]
    u_agg["user_recency_days"] = (cutoff_date - u_agg["last_date"]).dt.days.clip(lower=0)
    user_features = u_agg[
        ["user_total_orders", "user_total_qty", "user_avg_order_spend", "user_avg_item_price", "user_recency_days"]
    ].to_dict(orient="index")

    # 2. Item-level features
    i_agg = df.groupby("StockCode").agg(
        item_total_orders=("InvoiceNo", "nunique"),
        item_unique_buyers=("CustomerID", "nunique"),
        item_avg_price=("UnitPrice", "mean"),
        last_date=("InvoiceDate", "max"),
    )
    i_agg["item_recency_days"] = (cutoff_date - i_agg["last_date"]).dt.days.clip(lower=0)
    item_features = i_agg[
        ["item_total_orders", "item_unique_buyers", "item_avg_price", "item_recency_days"]
    ].to_dict(orient="index")

    # Item descriptions & global popularity list (for Baseline + Cold-Start Fallback)
    item_descriptions = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iloc[0]).to_dict()
    pop_df = i_agg.sort_values(["item_unique_buyers", "item_total_orders"], ascending=False)
    popular_items = [(code, int(row["item_unique_buyers"])) for code, row in pop_df.iterrows()]

    # 3. User-Item interaction history
    ui_agg = df.groupby(["CustomerID", "StockCode"]).agg(
        prev_qty=("Quantity", "sum"),
        last_date=("InvoiceDate", "max"),
    ).reset_index()
    ui_agg["recency_days"] = (cutoff_date - ui_agg["last_date"]).dt.days.clip(lower=0)

    user_item_history = {}
    for row in ui_agg.itertuples(index=False):
        user_item_history.setdefault(row.CustomerID, {})[row.StockCode] = (
            float(row.prev_qty),
            float(row.recency_days),
        )

    # 4. Stage 1 Retriever: Sparse Item-Item Cosine Similarity
    users = sorted(user_features.keys())
    items = sorted(item_features.keys())
    u_to_idx = {u: idx for idx, u in enumerate(users)}
    i_to_idx = {it: idx for idx, it in enumerate(items)}

    rows = ui_agg["CustomerID"].map(u_to_idx).values
    cols = ui_agg["StockCode"].map(i_to_idx).values
    vals = np.ones(len(ui_agg), dtype=np.float32)

    ui_matrix = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    sim_matrix = cosine_similarity(ui_matrix.T, dense_output=False)

    item_sim_top_k = {}
    for i_idx, code in enumerate(items):
        row_vec = sim_matrix.getrow(i_idx)
        indices = row_vec.indices
        scores = row_vec.data
        # Exclude self-similarity
        mask = indices != i_idx
        indices, scores = indices[mask], scores[mask]
        if len(scores) > top_neighbors:
            top_pos = np.argpartition(scores, -top_neighbors)[-top_neighbors:]
            indices, scores = indices[top_pos], scores[top_pos]
        order = np.argsort(scores)[::-1]
        item_sim_top_k[code] = [(items[indices[k]], float(scores[k])) for k in order if scores[k] > 0]

    return {
        "feature_names": FEATURE_COLUMNS,
        "user_features": user_features,
        "item_features": item_features,
        "user_item_history": user_item_history,
        "item_descriptions": item_descriptions,
        "popular_items": popular_items,
        "item_sim_top_k": item_sim_top_k,
        "cutoff_date": str(cutoff_date),
    }

def retrieve_candidates(user_id: str, store: dict, n_candidates: int = 50):
    """Stage 1: Retrieves top-N candidate items for a user via co-occurrence + popularity backfill."""
    history = store["user_item_history"].get(user_id)
    if not history:
        # Cold-start user
        return [(item, 0.0) for item, _ in store["popular_items"][:n_candidates]], True

    scores = {}
    sim_lookup = store["item_sim_top_k"]

    for bought_item, (prev_qty, _) in history.items():
        weight = 1.0 + np.log1p(prev_qty) * 0.25
        # Include repeat-purchase signal as a valid candidate
        scores[bought_item] = scores.get(bought_item, 0.0) + (0.5 * weight)
        for neighbor_item, sim_score in sim_lookup.get(bought_item, []):
            scores[neighbor_item] = scores.get(neighbor_item, 0.0) + (sim_score * weight)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_candidates]
    seen = {item for item, _ in ranked}

    # Backfill with globally popular items if fewer than n_candidates
    if len(ranked) < n_candidates:
        for pop_item, _ in store["popular_items"]:
            if pop_item not in seen:
                ranked.append((pop_item, 0.0))
                seen.add(pop_item)
                if len(ranked) >= n_candidates:
                    break

    return ranked, False

def extract_features_for_user(user_id: str, candidates: list, store: dict) -> pd.DataFrame:
    """Builds the exact feature matrix for Stage 2 LightGBM ranking."""
    u_feat = store["user_features"].get(user_id, {})
    u_hist = store["user_item_history"].get(user_id, {})
    i_feats = store["item_features"]

    u_orders = u_feat.get("user_total_orders", np.nan)
    u_qty = u_feat.get("user_total_qty", np.nan)
    u_avg_spend = u_feat.get("user_avg_order_spend", np.nan)
    u_avg_price = u_feat.get("user_avg_item_price", np.nan)
    u_recency = u_feat.get("user_recency_days", np.nan)

    rows = []
    for item_id, ret_score in candidates:
        it = i_feats.get(item_id, {})
        prev_qty, ui_recency = u_hist.get(item_id, (0.0, 999.0))
        item_price = it.get("item_avg_price", np.nan)

        if pd.notna(item_price) and pd.notna(u_avg_price) and u_avg_price > 0:
            price_ratio = item_price / u_avg_price
        else:
            price_ratio = np.nan

        rows.append([
            ret_score,
            1.0 if prev_qty > 0 else 0.0,
            prev_qty,
            ui_recency,
            u_orders,
            u_qty,
            u_avg_spend,
            u_recency,
            it.get("item_total_orders", np.nan),
            it.get("item_unique_buyers", np.nan),
            item_price,
            it.get("item_recency_days", np.nan),
            price_ratio,
        ])

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
