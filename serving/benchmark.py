import os
import json
import time
import random
import joblib
import httpx
import numpy as np
import matplotlib.pyplot as plt

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
STORE_PATH = "artifacts/feature_store.joblib"
OUT_JSON = "artifacts/benchmark_results.json"
OUT_PNG = "artifacts/latency_distribution.png"

def percentile_stats(latencies_ms: list) -> dict:
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }

def run_benchmark(n_warm: int = 250, n_cold: int = 50):
    store = joblib.load(STORE_PATH)
    known_users = list(store["user_features"].keys())
    rng = random.Random(42)

    warm_sample = [rng.choice(known_users) for _ in range(n_warm)]
    cold_sample = [f"COLD_USER_{i}" for i in range(n_cold)]

    # Verify service health & warm up JIT/caches with 5 requests
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        for u in warm_sample[:5]:
            client.post("/predict", json={"user_id": u, "top_k": 10})

        warm_latencies = []
        cold_latencies = []
        errors = 0

        t_start = time.perf_counter()

        for u in warm_sample:
            t0 = time.perf_counter()
            r = client.post("/predict", json={"user_id": u, "top_k": 10})
            elapsed = (time.perf_counter() - t0) * 1000.0
            if r.status_code == 200:
                warm_latencies.append(elapsed)
            else:
                errors += 1

        for u in cold_sample:
            t0 = time.perf_counter()
            r = client.post("/predict", json={"user_id": u, "top_k": 10})
            elapsed = (time.perf_counter() - t0) * 1000.0
            if r.status_code == 200:
                cold_latencies.append(elapsed)
            else:
                errors += 1

        total_sec = time.perf_counter() - t_start

    all_latencies = warm_latencies + cold_latencies
    total_reqs = len(all_latencies)
    throughput_rps = round(total_reqs / total_sec, 2) if total_sec > 0 else 0.0

    results = {
        "total_requests": total_reqs,
        "error_count": errors,
        "error_rate_pct": round((errors / max(total_reqs + errors, 1)) * 100.0, 4),
        "total_elapsed_sec": round(total_sec, 3),
        "throughput_req_per_sec": throughput_rps,
        "overall": percentile_stats(all_latencies),
        "warm_two_stage": percentile_stats(warm_latencies),
        "cold_start_fallback": percentile_stats(cold_latencies),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== EMPIRICAL API LATENCY & THROUGHPUT BENCHMARK ===")
    print(json.dumps(results, indent=2))

    # Plot latency distribution for PDF report
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.hist(warm_latencies, bins=30, alpha=0.75, label="Warm User (Two-Stage Ranker)", color="#2563eb")
    ax.hist(cold_latencies, bins=15, alpha=0.75, label="Cold-Start (Popularity Fallback)", color="#10b981")

    p50 = results["warm_two_stage"]["p50_ms"]
    p95 = results["warm_two_stage"]["p95_ms"]
    p99 = results["warm_two_stage"]["p99_ms"]
    ax.axvline(p50, color="#1e293b", linestyle="--", linewidth=1.2, label=f"Warm p50: {p50:.1f} ms")
    ax.axvline(p95, color="#f59e0b", linestyle="--", linewidth=1.2, label=f"Warm p95: {p95:.1f} ms")
    ax.axvline(p99, color="#ef4444", linestyle="--", linewidth=1.2, label=f"Warm p99: {p99:.1f} ms")

    ax.set_xlabel("End-to-End HTTP Latency (ms)")
    ax.set_ylabel("Request Count")
    ax.set_title(f"FastAPI Inference Latency Profile (n={total_reqs} requests, {throughput_rps} req/s)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"[SUCCESS] Saved {OUT_JSON} and {OUT_PNG}")

if __name__ == "__main__":
    run_benchmark()
