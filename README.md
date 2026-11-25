# Mini Production ML System: E-Commerce Two-Stage Recommendation & Ranking

End-to-end production machine learning system for Top-K e-commerce product recommendations built on the **UCI Online Retail** dataset. Implements leak-free temporal splitting, two-stage candidate retrieval and LightGBM ranking, a decoupled FastAPI REST service with Pydantic schema validation and graceful fallbacks, empirical latency profiling, and PSI/KS drift monitoring.

## Architecture Overview

1. **Data Pipeline (`src/data_pipeline.py`)**: Cleans raw transactions and enforces strict calendar-date cutoffs (**Train**: Dec 2010–Sep 2011, **Val**: Oct 2011, **Test**: Nov–Dec 2011) with automated assertions preventing look-ahead leakage.
2. **Stage 1 — Candidate Retrieval (`src/features.py`)**: Sparse item-to-item cosine similarity matrix fitted strictly on training interactions retrieves the top 50 candidate SKUs per user, backfilled with globally popular items.
3. **Stage 2 — Candidate Ranking (`src/train.py`)**: Scikit-Learn `Pipeline` (`SimpleImputer` -> `StandardScaler` -> class-weighted `LGBMClassifier`) trained on an internal temporal split (`< Aug 15, 2011` features, `>= Aug 15, 2011` labels) and serialized via `joblib`.
4. **Inference Service (`serving/app.py`, `serving/schemas.py`)**: FastAPI REST service exposing `/health` and `/predict` with Pydantic validation, cold-start routing (`cold_start_popularity`), and runtime exception fallback (`degraded_popularity_fallback`).
5. **Profiling & Monitoring (`serving/benchmark.py`, `src/drift_monitor.py`)**: Benchmarks p50/p95/p99 HTTP latency and computes PSI/KS feature drift + entity cold-start ratios.

## Repository Structure

```text
mini-production-ml-system/
├── README.md
├── requirements.txt
├── data/
│   └── fetch_data.py
├── artifacts/
│   ├── ranker_pipeline.joblib
│   ├── feature_store.joblib
│   ├── training_summary.json
│   ├── eval_results.json
│   ├── benchmark_results.json
│   ├── drift_report.json
│   ├── eval_comparison.png
│   ├── latency_distribution.png
│   └── system_architecture.png
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── drift_monitor.py
├── serving/
│   ├── app.py
│   ├── schemas.py
│   └── benchmark.py
└── tests/
    └── test_api.py
```

## Quickstart & Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Ingestion, Training & Offline Evaluation
```bash
python data/fetch_data.py
python src/data_pipeline.py
python src/train.py
python src/evaluate.py
python src/drift_monitor.py
```

### 3. Run Unit & Edge-Case Tests
```bash
pytest -v tests/test_api.py
```

### 4. Start the FastAPI Inference Service
```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

### 5. Sample `curl` Requests
```bash
# Health Check
curl -s http://127.0.0.1:8000/health | python -m json.tool

# Warm User Recommendation
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "12347", "top_k": 5, "exclude_items": ["85123A"]}' | python -m json.tool

# Cold-Start User (Automatic Popularity Fallback)
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "NEW_VISITOR_999", "top_k": 5}' | python -m json.tool
```

### 6. Run Empirical Latency Benchmark (300 Requests)
With `uvicorn` running in another terminal:
```bash
python serving/benchmark.py
```

## Key Empirical Results

| Metric (Test Set, Warm Users n=1440) | Popularity Baseline | Two-Stage Ranker | Relative Lift |
| :--- | :---: | :---: | :---: |
| **Precision@10** | 0.0801 | **0.2656** | +231.66% |
| **Recall@10** | 0.0307 | **0.1441** | +369.81% |
| **NDCG@10** | 0.0873 | **0.3095** | +254.61% |
| **HitRate@10** | 0.4806 | **0.7708** | +60.40% |

* **API Latency (300 requests, 0% errors):** Overall **p50 = 11.12 ms**, **p95 = 16.27 ms**, **p99 = 18.79 ms** | Throughput: **96.48 req/s**.
