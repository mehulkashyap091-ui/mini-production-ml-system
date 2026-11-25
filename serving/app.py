import os
import sys
import time
import logging
import warnings
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

# Ensure project root is on sys.path for clean modular imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import retrieve_candidates, extract_features_for_user
from serving.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
    HealthResponse,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("recsys-api")

PIPELINE_PATH = os.path.join(ROOT_DIR, "artifacts", "ranker_pipeline.joblib")
STORE_PATH = os.path.join(ROOT_DIR, "artifacts", "feature_store.joblib")

class ModelRegistry:
    pipeline = None
    store = None

registry = ModelRegistry()

def load_artifacts():
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(STORE_PATH):
        raise FileNotFoundError("Serialized artifacts missing. Run `python src/train.py` first.")
    registry.pipeline = joblib.load(PIPELINE_PATH)
    registry.store = joblib.load(STORE_PATH)
    logger.info("Loaded ranker pipeline and feature store successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="E-Commerce Two-Stage Recommendation & Ranking API",
    version="1.0.0",
    description="Production-grade two-stage retrieval + LightGBM ranking service with cold-start and degradation fallbacks.",
    lifespan=lifespan,
)

def _build_popularity_items(store: dict, top_k: int, exclude_set: set):
    items = []
    max_pop = store["popular_items"][0][1] if store["popular_items"] else 1
    for code, pop_count in store["popular_items"]:
        if code in exclude_set:
            continue
        items.append((code, round(pop_count / max_pop, 4)))
        if len(items) >= top_k:
            break
    return items

@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded = registry.pipeline is not None and registry.store is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    store = registry.store
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=store["metadata"]["model_version"],
        catalog_items=len(store["item_features"]),
        known_users=len(store["user_features"]),
    )

@app.post("/predict", response_model=RecommendationResponse)
def predict(request: RecommendationRequest):
    t0 = time.perf_counter()
    if registry.pipeline is None or registry.store is None:
        raise HTTPException(status_code=503, detail="Inference service unavailable: artifacts not loaded")

    store = registry.store
    exclude_set = set(request.exclude_items or [])
    descriptions = store["item_descriptions"]
    model_version = store["metadata"]["model_version"]

    # Stage 1: Candidate Retrieval
    n_fetch = max(50, request.top_k + len(exclude_set))
    candidates, is_cold_start = retrieve_candidates(request.user_id, store, n_candidates=n_fetch)

    fallback_used = False
    if is_cold_start:
        strategy = "cold_start_popularity"
        fallback_used = True
        ranked = _build_popularity_items(store, request.top_k, exclude_set)
    else:
        try:
            # Filter excluded items before Stage 2 scoring
            filtered_cands = [(code, sc) for code, sc in candidates if code not in exclude_set]
            if not filtered_cands:
                raise ValueError("No eligible candidates left after exclusions")

            # Stage 2: LightGBM Ranking
            X_cand = extract_features_for_user(request.user_id, filtered_cands, store)
            probs = registry.pipeline.predict_proba(X_cand)[:, 1]
            if np.isnan(probs).any():
                raise ValueError("Ranker produced NaN probability scores")

            order = np.argsort(probs)[::-1][: request.top_k]
            ranked = [(filtered_cands[i][0], round(float(probs[i]), 4)) for i in order]
            strategy = "two_stage_ranker"
        except Exception as exc:
            # Graceful degradation fallback
            logger.error(f"Stage 2 ranking failed for user={request.user_id}: {exc}. Using popularity fallback.")
            strategy = "degraded_popularity_fallback"
            fallback_used = True
            ranked = _build_popularity_items(store, request.top_k, exclude_set)

    recs = [
        RecommendedItem(
            rank=idx + 1,
            stock_code=code,
            description=descriptions.get(code, "UNKNOWN"),
            score=score,
        )
        for idx, (code, score) in enumerate(ranked)
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return RecommendationResponse(
        user_id=request.user_id,
        top_k=len(recs),
        strategy=strategy,
        is_cold_start=is_cold_start,
        fallback_used=fallback_used,
        latency_ms=latency_ms,
        model_version=model_version,
        recommendations=recs,
    )
