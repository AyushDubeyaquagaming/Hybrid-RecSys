"""Prometheus metric definitions for the BetBlitz RecSys API.

All metrics are defined here and imported by main.py and routes/recommendations.py.
Centralising them prevents duplicate registration errors on hot reload.
"""
from prometheus_client import Counter, Gauge, Histogram

RECOMMEND_REQUESTS = Counter(
    "recsys_recommend_requests_total",
    "Total recommendation requests",
    ["source"],  # "lightfm" or "popularity_fallback"
)

RECOMMEND_LATENCY = Histogram(
    "recsys_recommend_latency_seconds",
    "Recommendation latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

COLD_START_TOTAL = Counter(
    "recsys_cold_start_total",
    "Cold-start popularity fallback count",
)

ITEMS_RETURNED = Histogram(
    "recsys_items_returned",
    "Number of items returned per recommendation request",
    buckets=[0, 1, 2, 3, 4, 5, 10, 20],
)

MODEL_LOADED = Gauge(
    "recsys_model_loaded",
    "1 if the LightFM model is loaded and serving, 0 otherwise",
)
