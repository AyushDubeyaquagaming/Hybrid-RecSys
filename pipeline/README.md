# Training Pipeline

Prefect-orchestrated training pipeline that ingests live casino event data from MongoDB, builds a LightFM hybrid recommendation model, evaluates it, exports model artifacts, and optionally publishes embeddings to Neo4j.

---

## Overview

The pipeline runs as a single Prefect flow (`pipeline.flow`) with ten ordered stages. Each stage is a Prefect `@task` that can be individually retried or skipped depending on configuration flags. The pipeline is designed to be triggered nightly via `make retrain`.

---

## Pipeline Flow

| Stage | Task module | Input | Output | Description |
|---|---|---|---|---|
| 1 | `steps.ingest` | MongoDB | Raw DataFrames | Load transactions, game details, users |
| 2 | `steps.enrich` | Raw DataFrames | Enriched DataFrames | Session + device attribution |
| 3 | `steps.align` | Enriched DataFrames | Canonical DataFrame | Unified schema alignment |
| 4 | `steps.build_dataset` | Canonical DataFrame | Dataset artifacts (matrices, maps) | Temporal split, LightFM dataset construction, feature matrices |
| 5 | `steps.train` | Dataset artifacts | Trained model | LightFM WARP training loop |
| 6 | `steps.evaluate` | Model + test set | Metrics dict | Precision@K, AUC, NDCG |
| 6b | `steps.diagnostics` | Model + dataset | Plot files | Diagnostic plots (logged to MLflow) |
| 7 | `steps.export` | Model + artifacts | Disk files | Serialise artifacts, write Redis feature hashmaps |
| 8 | MLflow registry | Model | MLflow run URI | Register pyfunc model in MLflow Model Registry |
| 9 | Redis | Feature tables | Redis hashmaps | User and game features with TTL |
| 10 | `steps.neo4j_export` | Model + maps | Neo4j nodes | Push embeddings to Neo4j Player/Game nodes (best-effort) |

---

## File Structure

```
pipeline/
├── config.py           # PipelineSettings — Pydantic BaseSettings, fully env-var driven
├── flow.py             # Prefect flow: run_training_flow()
├── run.py              # CLI entrypoint: python -m pipeline.run
├── db.py               # Shared MongoDB connection helper
├── exceptions.py       # Custom pipeline exceptions
├── logging_utils.py    # Structured logging setup
├── mlflow_pyfunc.py    # Custom MLflow pyfunc model wrapper
├── requirements.txt    # Python dependencies for this package
└── steps/
    ├── ingest.py           # MongoDB → pandas
    ├── enrich.py           # Session and device enrichment
    ├── align.py            # Canonical event schema
    ├── features.py         # Implicit score, interaction matrix, feature tables
    ├── build_dataset.py    # LightFM dataset + temporal train/test split
    ├── train.py            # LightFM WARP training
    ├── evaluate.py         # Precision@K, AUC, NDCG
    ├── diagnostics.py      # Diagnostic plot generation
    ├── export.py           # Disk + Redis feature export
    └── neo4j_export.py     # Neo4j embedding export (best-effort)
```

---

## Configuration

All settings are loaded from environment variables (or a `.env` file in the repo root) via `PipelineSettings` in `config.py`.

### MongoDB

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `booktestdbgp_test` | Source database name |
| `MONGO_DIRECT_CONNECTION` | `true` | Connect directly (bypass replica set routing) |
| `MONGO_TIMEOUT_MS` | `10000` | Connection timeout in milliseconds |

### Model Hyperparameters

| Variable | Default | Description |
|---|---|---|
| `NO_COMPONENTS` | `32` | Embedding dimensionality |
| `LOSS` | `warp` | LightFM loss function |
| `LEARNING_RATE` | `0.03` | SGD learning rate |
| `ITEM_ALPHA` | `1e-6` | Item L2 regularisation |
| `USER_ALPHA` | `1e-6` | User L2 regularisation |
| `N_EPOCHS` | `20` | Training epochs |
| `SEED` | `42` | Random seed |
| `EVAL_K` | `5` | K for Precision@K and NDCG@K |

### MLflow

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_ENABLED` | `false` | Enable MLflow experiment tracking |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow tracking server URI |
| `MLFLOW_EXPERIMENT_NAME` | `betblitz-recsys` | Experiment name |
| `MLFLOW_REGISTRY_ENABLED` | `true` | Register model in MLflow Model Registry |
| `MLFLOW_REGISTERED_MODEL_NAME` | `betblitz-lightfm-recsys` | Registered model name |
| `EDA_ENABLED` | `true` | Log EDA diagnostic plots to MLflow |

### Redis Feature Store

| Variable | Default | Description |
|---|---|---|
| `REDIS_ENABLED` | `false` | Enable Redis feature export |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `""` | Redis password |
| `REDIS_TTL_SECONDS` | `90000` | Key TTL (~25 hours, covers 1 nightly cycle) |

### Neo4j Embedding Store

| Variable | Default | Description |
|---|---|---|
| `NEO4J_ENABLED` | `false` | Enable Neo4j embedding export during pipeline run |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `""` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Target Neo4j database |
| `NEO4J_PLAYER_KEY` | `id` | Property name used to match Player nodes |
| `NEO4J_GAME_KEY` | `id` | Property name used to match Game nodes |
| `NEO4J_BATCH_SIZE` | `500` | Embedding batch size per Cypher write |

### Output

| Variable | Default | Description |
|---|---|---|
| `ARTIFACT_DIR` | `../betblitz-recsys-api/artifacts` | Directory where artifacts are written |

---

## Running the Pipeline

### Via Make (recommended)

```bash
# Ensure Docker services are up (MLflow, Redis)
make demo-up

# Run full training pipeline
make retrain
```

### Standalone (no MLflow, no Redis)

```bash
source venv/bin/activate

MLFLOW_ENABLED=false REDIS_ENABLED=false python -m pipeline.run
```

### With Neo4j enabled during training

```bash
NEO4J_ENABLED=true \
NEO4J_URI=bolt://<host>:7687 \
NEO4J_PASSWORD=<password> \
python -m pipeline.run
```

### Push embeddings post-training (without retraining)

```bash
make push-neo4j
```

This runs `scripts/push_neo4j_embeddings.py` which loads existing artifacts from `ARTIFACT_DIR` and pushes embeddings to Neo4j without triggering a full retrain.

---

## Artifacts Produced

All artifacts are written to `ARTIFACT_DIR` (default: `betblitz-recsys-api/artifacts/`).

| File | Description |
|---|---|
| `model.joblib` | Trained LightFM model |
| `dataset.joblib` | LightFM dataset object (user/item mappings) |
| `interactions.joblib` | Sparse interaction matrix |
| `user_features_matrix.joblib` | Sparse user feature matrix |
| `item_features_matrix.joblib` | Sparse item feature matrix |
| `user_id_map.json` | `{internal_index → external_user_id}` mapping |
| `item_id_map.json` | `{internal_index → external_game_id}` mapping |
| `game_metadata.json` | Game display metadata (name, type, vendor) |
| `popularity_ranking.json` | Popularity-ordered game list for cold-start |

---

## Neo4j Export Details

`steps/neo4j_export.py` writes LightFM embeddings to Neo4j after each training run. The export is best-effort — failures are caught and logged without failing the pipeline.

**Cypher (per batch):**

```cypher
UNWIND $batch AS row
MATCH (p:Player { id: row.id })
SET p.embedding = row.embedding,
    p.bias      = row.bias,
    p.updated_at = datetime()
RETURN count(p) AS matched
```

**Matching keys:**
- `Player` nodes matched on the property defined by `NEO4J_PLAYER_KEY` (default: `id`)
- `Game` nodes matched on the property defined by `NEO4J_GAME_KEY` (default: `id`)

Unmatched nodes are silently skipped; the task logs the number of matched nodes per entity type.

---

## Testing

```bash
# All pipeline tests
pytest tests/pipeline/ -v

# Individual test file
pytest tests/pipeline/test_neo4j_export.py -v
```

The test suite uses mocked MongoDB, MLflow, Redis, and Neo4j drivers — no live infrastructure is required. All 101 pipeline tests should pass.
