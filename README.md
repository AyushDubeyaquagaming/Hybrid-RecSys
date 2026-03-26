# BetBlitz — Live Casino Recommendation System

A production-grade hybrid recommendation system for live casino games, built on [LightFM](https://github.com/lyst/lightfm). The system trains nightly from MongoDB event data, exposes recommendations over a FastAPI service, and publishes learned embeddings to Neo4j for downstream graph-based retrieval.

---

## Architecture Overview

```
MongoDB
  │
  ▼
pipeline/           ← Prefect training pipeline (runs on host)
  │ model artifacts
  ├──────────────────► betblitz-recsys-api/   ← FastAPI serving layer (Docker)
  │ embeddings
  └──────────────────► Neo4j                  ← Graph embedding store
                                                (consumed by downstream services)

Observability
  ├── MLflow         ← Experiment tracking + model registry (Docker)
  ├── Prometheus     ← Metrics scraping (Docker)
  └── Grafana        ← Dashboards (Docker)
```

| Component | Role |
|---|---|
| `pipeline/` | Nightly Prefect training pipeline — ingests, trains, evaluates, exports |
| `betblitz-recsys-api/` | FastAPI service — serves real-time game recommendations |
| `scripts/push_neo4j_embeddings.py` | Standalone Neo4j embedding push utility |
| MLflow | Experiment tracking and model registry |
| Redis | Best-effort feature store for user and game features |
| Neo4j | Graph store enriched with LightFM embeddings after each training run |
| Prometheus + Grafana | API metrics and latency dashboards |

---

## Repository Structure

```
lightfm_project/
├── pipeline/                    # Training pipeline
│   ├── config.py                # All runtime settings (Pydantic, env-var driven)
│   ├── flow.py                  # Prefect flow — orchestrates all training stages
│   ├── run.py                   # Entrypoint: python -m pipeline.run
│   ├── steps/
│   │   ├── ingest.py            # MongoDB → pandas ingestion
│   │   ├── enrich.py            # Session and device enrichment
│   │   ├── align.py             # Canonical schema alignment
│   │   ├── features.py          # Feature engineering (implicit scores, user/game tables)
│   │   ├── build_dataset.py     # LightFM dataset construction + temporal split
│   │   ├── train.py             # LightFM WARP training loop
│   │   ├── evaluate.py          # Precision@K, AUC, NDCG evaluation
│   │   ├── diagnostics.py       # Diagnostic plot generation (MLflow artifacts)
│   │   ├── export.py            # Artifact export to disk + Redis feature write
│   │   └── neo4j_export.py      # Embedding export to Neo4j (best-effort, post-training)
│   └── requirements.txt
│
├── betblitz-recsys-api/         # FastAPI recommendation service
│   ├── app/
│   │   ├── main.py              # FastAPI app, lifespan artifact loading
│   │   ├── config.py            # Service-level settings
│   │   ├── metrics.py           # Prometheus metric definitions
│   │   ├── routes/
│   │   │   ├── recommendations.py  # POST /recommend
│   │   │   └── health.py           # GET /health
│   │   ├── services/
│   │   │   └── model_service.py    # Artifact loading, inference, cold-start fallback
│   │   └── schemas/
│   │       └── recommendation.py   # Pydantic request/response models
│   ├── artifacts/               # Model artifacts (runtime-mounted, not committed)
│   ├── Dockerfile
│   └── requirements.txt
│
├── scripts/
│   ├── push_neo4j_embeddings.py # Load artifacts from disk, push embeddings to Neo4j
│   └── smoke_test.sh            # End-to-end API smoke test
│
├── tests/
│   └── pipeline/                # Pipeline unit tests
│
├── monitoring/
│   ├── prometheus.yml           # Prometheus scrape config
│   └── grafana/                 # Grafana provisioning and dashboard JSON
│
├── docker-compose.yml           # Full local stack
└── Makefile                     # Operational commands
```

---

## Prerequisites

- Docker and Docker Compose
- Python 3.10
- Access to the MongoDB instance (credentials via environment or `.env`)
- Neo4j instance credentials (for embedding export)

---

## Quick Start

### 1. Set up the virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r pipeline/requirements.txt
```

### 2. Configure environment

Export the following variables or add them to a `.env` file in the repo root:

```bash
# MongoDB
MONGO_URI=mongodb://<host>:27017
MONGO_DB=<database>

# Neo4j (required for make push-neo4j)
NEO4J_URI=bolt://<host>:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j
```

MLflow and Redis credentials are injected automatically by `make retrain`.

### 3. Start the full local stack

```bash
make demo-up
```

Starts: API, MLflow, Redis, Prometheus, Grafana. The command waits for all services to be healthy before returning.

### 4. Verify the stack

```bash
make demo-check
```

### 5. Run the training pipeline

```bash
make retrain
```

Runs the Prefect training flow on the host, connecting to Dockerized MLflow and Redis. On completion, artifacts are written to `betblitz-recsys-api/artifacts/`.

### 6. Push embeddings to Neo4j

```bash
export NEO4J_URI='bolt://<host>:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='<password>'
export NEO4J_DATABASE='neo4j'

make push-neo4j
```

### 7. Restart the API to serve fresh artifacts

```bash
make restart-api
```

---

## Training Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| 1 | `ingest` | Load transactions, game details, users from MongoDB |
| 2 | `enrich` | Session enrichment, device attribution |
| 3 | `align` | Canonical event schema alignment |
| 4 | `build_dataset` | Temporal train/test split, LightFM dataset and feature matrix construction |
| 5 | `train` | LightFM WARP training (32-dimensional embeddings) |
| 6 | `evaluate` | Precision@5, AUC, NDCG |
| 6b | `diagnostics` | Diagnostic plots logged as MLflow artifacts |
| 7 | `export` | Artifact serialisation to disk, user/game features to Redis |
| 8 | MLflow registry | Log pyfunc model, register in MLflow Model Registry |
| 9 | Redis | User and game feature HashMaps with TTL |
| 10 | `neo4j_export` | Embedding vectors to Neo4j Player/Game nodes (best-effort) |

---

## Recommendation Service

The FastAPI service loads pre-trained artifacts at startup and serves recommendations with sub-10ms median latency.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health, Redis status, model metadata |
| `POST` | `/recommend` | Return top-K game recommendations for a player |

**Inference:**
- Known users → LightFM hybrid inference using user and item feature matrices, excluding already-played games
- Unknown users → Popularity-ranked cold-start fallback

See [betblitz-recsys-api/README.md](betblitz-recsys-api/README.md) for full API usage.

---

## Neo4j Embedding Store

After each training run, `make push-neo4j` loads the latest artifacts from disk and publishes LightFM embedding vectors to Neo4j. These embeddings can be used by downstream services for graph-based candidate retrieval.

**Graph schema:**

```cypher
(:Player { id: STRING, embedding: LIST<FLOAT>, bias: FLOAT, updated_at: DATETIME })
(:Game   { id: STRING, embedding: LIST<FLOAT>, bias: FLOAT, updated_at: DATETIME })
```

**Post-push verification:**

```cypher
MATCH (p:Player) WHERE p.embedding IS NOT NULL RETURN count(p) AS players_with_embeddings
MATCH (g:Game)   WHERE g.embedding IS NOT NULL RETURN count(g) AS games_with_embeddings
```

---

## Observability

| Service | URL | Default credentials |
|---|---|---|
| API Swagger | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

The Grafana dashboard tracks request rate, error rate, recommendation latency (p50/p95/p99), items returned, outcome distribution, and session duration coverage.

---

## Makefile Reference

| Command | Description |
|---|---|
| `make demo-up` | Start all Docker services |
| `make demo-down` | Stop all Docker services |
| `make retrain` | Run full training pipeline (Docker services must be running) |
| `make push-neo4j` | Push embeddings to Neo4j from latest artifacts |
| `make demo-check` | Run API smoke tests |
| `make restart-api` | Restart the API container to load fresh artifacts |
| `make logs` | Tail Docker service logs |

---

## Testing

```bash
# Pipeline unit tests
pytest tests/pipeline/ -v

# API tests
pytest betblitz-recsys-api/tests/ -v
```

CI runs both suites on every push and pull request to `master`.

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML framework | LightFM 1.17 (WARP loss, hybrid collaborative + content-based) |
| Pipeline orchestration | Prefect 3 |
| Data source | MongoDB (pymongo) |
| Serving | FastAPI + Uvicorn |
| Experiment tracking | MLflow 2.10 |
| Feature store | Redis 7 |
| Graph store | Neo4j 5 |
| Observability | Prometheus + Grafana |
| Configuration | Pydantic Settings (env-var driven) |
| Testing | pytest |
| Infrastructure | Docker Compose |
