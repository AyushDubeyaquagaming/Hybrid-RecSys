# BetBlitz Recommendation API

FastAPI service that loads pre-trained LightFM artifacts at startup and serves real-time live casino game recommendations. Supports hybrid inference for known users and popularity-based cold-start for new players.

---

## Overview

| Attribute | Value |
|---|---|
| Framework | FastAPI + Uvicorn |
| Port | 8000 |
| Artifact loading | On startup (lifespan) |
| Known-user inference | LightFM hybrid scoring, excludes already-played games |
| Cold-start fallback | Popularity ranking |
| Metrics | Prometheus (`/metrics`) |

---

## Startup

### Via Docker Compose (recommended)

```bash
# From the repo root
make demo-up
```

The API starts as part of the full stack. Artifacts are mounted from `betblitz-recsys-api/artifacts/` at container startup.

### Standalone Docker

```bash
cd betblitz-recsys-api

docker build -t betblitz-recsys-api .

docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  --name betblitz-api \
  betblitz-recsys-api
```

### Local (without Docker)

```bash
source ../venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

> Artifacts must exist in `./artifacts/` before starting the server. Run `make retrain` from the repo root to generate them.

---

## API Reference

### `GET /health`

Returns service health and model metadata.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "model_version": "v1.0",
  "n_users": 4321,
  "n_items": 150
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"healthy"` or `"degraded"` |
| `model_loaded` | boolean | Whether artifacts loaded successfully |
| `redis_connected` | boolean | Redis feature store availability (best-effort) |
| `n_users` | integer | Number of users in the trained model |
| `n_items` | integer | Number of games in the trained model |

---

### `POST /recommend`

Return ranked game recommendations for a player.

**Request body:**

```json
{
  "user_id": "5097103780",
  "top_k": 5,
  "exclude_played": true
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | string | Yes | — | External player identifier |
| `top_k` | integer | No | `5` | Number of recommendations to return (max: `MAX_TOP_K`) |
| `exclude_played` | boolean | No | `true` | Exclude games the user has already interacted with |

**Response — known user:**

```json
{
  "user_id": "5097103780",
  "recommendations": [
    { "game_id": "game_42", "score": 0.934, "rank": 1 },
    { "game_id": "game_07", "score": 0.891, "rank": 2 }
  ],
  "is_cold_start": false,
  "top_k": 2
}
```

**Response — cold-start (unknown user):**

```json
{
  "user_id": "new_player_999",
  "recommendations": [
    { "game_id": "game_11", "score": null, "rank": 1 },
    { "game_id": "game_03", "score": null, "rank": 2 }
  ],
  "is_cold_start": true,
  "top_k": 2
}
```

---

## Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Recommendations for a known user
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "5097103780", "top_k": 5}'

# Cold-start user
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "unknown_player"}'

# Include already-played games
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "5097103780", "top_k": 10, "exclude_played": false}'
```

Interactive Swagger UI is available at `http://localhost:8000/docs`.

---

## Artifact Contract

The service expects the following files in `ARTIFACT_DIR` at startup.

| File | Description |
|---|---|
| `model.joblib` | Trained LightFM model |
| `dataset.joblib` | LightFM dataset object (user/item mappings) |
| `user_features_matrix.joblib` | Sparse user feature matrix |
| `item_features_matrix.joblib` | Sparse item feature matrix |
| `user_id_map.json` | `{internal_index → external_user_id}` |
| `item_id_map.json` | `{internal_index → external_game_id}` |
| `game_metadata.json` | Game display metadata (name, type, vendor) |
| `popularity_ranking.json` | Popularity-ordered game list for cold-start |

Generate all artifacts by running `make retrain` from the repo root.

---

## Prometheus Metrics

Available at `GET /metrics` (Prometheus scrape endpoint).

| Metric | Type | Description |
|---|---|---|
| `recommendations_total` | Counter | Total recommendation requests |
| `recommendations_errors_total` | Counter | Failed recommendation requests |
| `recommendation_latency_seconds` | Histogram | End-to-end request latency |
| `recommendation_items_returned` | Histogram | Number of items returned per request |
| `cold_start_total` | Counter | Cold-start fallback invocations |

---

## Configuration

All settings are loaded from environment variables.

| Variable | Default | Description |
|---|---|---|
| `ARTIFACT_DIR` | `./artifacts` | Path to model artifact directory |
| `DEFAULT_TOP_K` | `5` | Default number of recommendations |
| `MAX_TOP_K` | `20` | Maximum allowed `top_k` value |
| `PREDICT_NUM_THREADS` | `1` | LightFM predict parallelism |
| `REDIS_HOST` | `redis` | Redis hostname (feature store, best-effort) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `""` | Redis password |

---

## Testing

```bash
# From the repo root
pytest betblitz-recsys-api/tests/ -v
```

Tests cover: health endpoint, recommendation endpoint, cold-start path, model service unit tests, and Redis health integration. No live infrastructure required — all external dependencies are mocked.
