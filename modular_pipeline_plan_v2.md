# BetBlitz Modular ML Pipeline — Implementation Plan (v0.2)

---

## Context for Claude Code

You are converting a working Jupyter notebook (`lightfm/recsys_real_data.ipynb`) into a modular Python pipeline orchestrated by **Prefect**. The notebook trains a LightFM recommendation model for BetBlitz, an iGaming platform.

**The goal is NOT to improve the model or add new features.** The goal is to take the exact same logic from the notebook and organize it into runnable, testable, maintainable Python modules that Prefect can orchestrate as a nightly flow.

The pipeline outputs the **same artifact bundle** that the existing FastAPI service (`betblitz-recsys-api/`) already consumes. Nothing changes on the serving side.

**Primary engineering objective for v0.2:** preserve notebook behavior, keep artifact compatibility with the current FastAPI service, and make the training/export path reliable enough for scheduled execution. Accuracy improvement is explicitly out of scope.

---

## 1. Where It Lives

Same repo. The directory sits alongside the existing code:

```
betblitz-recsys/                      # repo root
├── betblitz-recsys-api/              # FastAPI serving (already built, do not touch)
│   ├── app/
│   ├── artifacts/                    # ← pipeline writes here
│   ├── scripts/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── lightfm/                          # notebooks (reference only, do not modify)
│   ├── recsys.ipynb
│   └── recsys_real_data.ipynb
├── pipeline/                         # ← NEW: modular ML pipeline
│   ├── __init__.py
│   ├── config.py                     # All settings: MongoDB, paths, model hyperparams
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── ingest.py                 # MongoDB data loading + cleaning
│   │   ├── enrich.py                 # Session duration + device type joins
│   │   ├── align.py                  # TDD schema alignment (map_* functions)
│   │   ├── features.py               # Feature engineering (event, user, game, user-game)
│   │   ├── build_dataset.py          # LightFM Dataset, split, interactions, feature matrices
│   │   ├── train.py                  # LightFM training
│   │   ├── evaluate.py               # Metrics computation (P@5, AUC, NDCG)
│   │   └── export.py                 # Artifact export to betblitz-recsys-api/artifacts/
│   ├── flow.py                       # Prefect flow: wires all steps together
│   ├── run.py                        # CLI entry point: python -m pipeline.run
│   └── requirements.txt              # Pipeline-specific dependencies
├── tests/
│   └── pipeline/
│       ├── __init__.py
│       ├── test_ingest.py
│       ├── test_align.py
│       ├── test_features.py
│       ├── test_build_dataset.py
│       ├── test_train.py
│       └── test_artifact_contract.py
├── .gitignore
└── README.md
```

---

## 2. Config (`pipeline/config.py`)

Single source of truth for all settings. Uses `pydantic-settings` with `.env` override.

```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    # MongoDB
    MONGO_URI: str = "mongodb://100.103.177.85:27017"
    MONGO_DB: str = "booktestdbgp_test"
    MONGO_DIRECT_CONNECTION: bool = True
    MONGO_TIMEOUT_MS: int = 10000

    # Data filters
    GAME_TYPE_FILTER: str = "Live Casino"
    BET_STATUS_FILTER: str = "SETTLED"
    SESSION_SOURCE_FILTER: str = "livecasino"
    SESSION_JOIN_TOLERANCE_MIN: int = 60
    DEVICE_JOIN_TOLERANCE_HOURS: int = 24

    # Game name normalization
    GAME_NAME_MAP: dict = {
        "998:baccarat": "Baccarat",
        "Football studio": "Football Studio",
    }

    # Model hyperparameters
    NO_COMPONENTS: int = 32
    LOSS: str = "warp"
    LEARNING_RATE: float = 0.03
    ITEM_ALPHA: float = 1e-6
    USER_ALPHA: float = 1e-6
    N_EPOCHS: int = 20
    SEED: int = 42
    PREDICT_NUM_THREADS: int = 4

    # Evaluation
    EVAL_K: int = 5

    # Output
    ARTIFACT_DIR: str = str(
        Path(__file__).resolve().parent.parent / "betblitz-recsys-api" / "artifacts"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```

Notes:
- Use a repo-root-derived artifact path, not a cwd-relative path. This avoids path bugs under Prefect scheduling.
- Do **not** introduce new config flags that alter notebook behavior unless they are explicitly intended for future divergence.

---

## 3. Pipeline Steps — What Goes Where

Each step is a **Prefect task** (decorated with `@task`). Each takes explicit inputs, returns explicit outputs. No global state.

### `steps/ingest.py` — Data Loading & Cleaning

**Notebook cells:** 1, 2, 3, 4, 5, 6, 8

```
@task
def load_transactions(settings) -> pd.DataFrame:
    Connect to MongoDB
    Load bet_transactions (Live Casino, SETTLED, betParts exists)
    Extract betParts[0] → sportName, categoryName, tournamentName, tournamentId
    Rename fields: loginId→userId, stake→betAmount, createdDate→timestamp,
                   sportName→gameName, tournamentName→providerName_raw
    Add time features: hourOfDay, dayOfWeek
    Drop rows with critical nulls (userId, gameName, betAmount)
    Return transactions_df

@task
def load_game_details(settings) -> pd.DataFrame:
    Load gamedetails (gameStatus=ON)
    Extract category[0].name → gameType
    Return games_df (gameId, gameName, gameType, gamevendor, minBet, maxBet)

@task
def load_users(settings) -> pd.DataFrame:
    Load players (activeStatus=True)
    Rename contactNo → userId
    Return users_df (userId, playerId, activeStatus)

@task
def clean_and_merge(transactions_df, games_df, settings) -> pd.DataFrame:
    Normalize game names (settings.GAME_NAME_MAP)
    Merge gameId + gamevendor from games_df (deduplicated on gameName)
    Set providerName = providerName_raw.combine_first(gamevendor)
    Add binary win flag
    Return cleaned transactions_df
```

### `steps/enrich.py` — Session & Device Enrichment

**Notebook cells:** 14, 15

These two steps must preserve notebook behavior. Do **not** simplify them just because the code is awkward. This part directly affects `durationSeconds`, `entryPoint`, `deviceType`, `exitType`, implicit scores, and ultimately exported recommendations.

```
@task
def enrich_sessions(transactions_df, settings) -> pd.DataFrame:
    Mirror notebook cell 14 exactly.

    Required behavior:
      - Load usersessionlog with filter: status=Closed, source=livecasino
      - Compute session_duration_sec = updated_at - created_at in seconds
      - Do NOT filter out sessions under 30 seconds
      - Keep quick-exit candidates; notebook explicitly retains them
      - Normalize userId and gameId as strings
      - Create transactions_df['gameId_str'] exactly as in notebook
      - Run a single merge_asof keyed by:
          left_on='timestamp'
          right_on='updated_at'
          left_by=['userId', 'gameId_str']
          right_by=['userId', 'gameId']
          direction='backward'
          tolerance=60 minutes
      - Populate session_duration_sec and entryPoint_raw from the matched session
      - Drop temporary helper columns like gameId_str

    Important:
      - Do NOT add a userId-only fallback merge
      - Do NOT pre-filter sessions with a 30-second bounce threshold
      - The implementation goal here is notebook parity, not behavioral cleanup

@task
def enrich_device(transactions_df, users_df, settings) -> pd.DataFrame:
    Mirror notebook cell 15 exactly.

    Required behavior:
      - Load useractivitylogs with the same projected columns as the notebook
      - Detect timestamp column using the notebook priority order:
          updated_at, created_at, timestamp, event_time
      - Build player bridge from users_df: userId -> playerId_bridge
      - Preserve the candidate-join cascade exactly as the notebook:
          1. userId -> activity.playerId
          2. playerId_bridge -> activity.playerId
          3. userId -> activity.loginId
          4. userId -> activity.user_id
          5. userId -> activity.userId
      - For each candidate join:
          use backward merge_asof
          use 24-hour tolerance
          fill only rows where deviceType_raw is still missing
      - Return transactions_df with deviceType_raw

    Important:
      - Do NOT collapse this into a single detected user key plus one fallback
      - The ordered candidate-join loop is part of the notebook logic
```

### `steps/align.py` — TDD Schema Alignment

**Notebook cells:** 22, 24

```
# Pure functions (no @task needed, called inside the align task):
is_objectid_like(raw_value) -> bool
map_game_type(raw_value) -> str
map_provider(raw_value) -> str
map_device_type(raw_value) -> str
map_entry_point(raw_value) -> str
map_time_of_day(ts: pd.Series) -> pd.Series
map_day_of_week(ts: pd.Series) -> pd.Series
map_outcome(raw_result: pd.Series) -> pd.Series

@task
def align_to_schema(transactions_df) -> pd.DataFrame:
    Build events_df with all 17 SCHEMA_COLUMNS
    Apply all map_* functions
    ID sanitization (strip, null guard)
    Compute returnedWithin10mins, exitType
    Compute timeOfDay, dayOfWeek
    Drop invalid rows
    Return events_df
```

**IMPORTANT:** The map functions are identical to the notebook. Copy them exactly. Particularly:
- `map_outcome` uses `value.isin(['LOSS', 'LOSE'])` (handles both spellings)
- `map_provider` uses `is_objectid_like()` to catch hex ObjectIds
- `map_game_type` handles `sic bo` (with space), `sicbo`, and crash aliases
- `entryPoint` remains semi-open to preserve real source values

### `steps/features.py` — Feature Engineering

**Notebook cells:** 26, 27, 29, 30, and the `build_feature_tables_from_events` helper from cell 34

```
def _mode_or_default(series, default_value):
    Notebook helper used in the dataset-building stage.

def build_feature_tables_from_events(base_events: pd.DataFrame) -> tuple:
    This is the notebook helper copied as-is.
    Returns:
      (fe_events, user_features_df, game_features_df, provider_features_df, user_game_df)

@task
def build_feature_tables(events_df) -> tuple:
    Thin task wrapper around build_feature_tables_from_events(events_df)
```

Internal logic:
- Event-level: `is_quick_exit`, `is_positive_outcome`, `engagement_intensity`, `minutes_since_prev_event`
- User-level: `total_sessions`, `unique_games`, `unique_providers`, `quick_exit_rate`, `return_10m_rate`, `positive_outcome_rate`, `preferred_time_of_day`, `preferred_day_of_week`, `preferred_device`, `preferred_entry_point`, `recency_days`
- Game-level: `game_sessions`, `unique_users`, `game_type`, `provider`, `popularity_score`
- User-game: `interaction_count`, `implicit_score = 0.40 freq + 0.25 engagement + 0.15 quality + 0.20 recency`

### `steps/build_dataset.py` — LightFM Dataset Construction

**Notebook cell:** 34 (split logic, dataset construction, feature matrices)

```
@task
def build_lightfm_dataset(events_df, games_df, settings) -> dict:
    Temporal holdout split:
      - For users with 2+ distinct games, hold out last-touched game
      - Fallback: one global holdout pair if too sparse

    Build train feature tables using build_feature_tables_from_events() on train events only
    Build item feature table (ife) with static metadata fallback for test-only items

    Token functions (copy exactly from notebook):
      user_tokens(row) → [preferred_time_of_day, preferred_day_of_week,
                          preferred_device, preferred_entry_point]
      item_tokens(row) → [game_type, provider, popularity_bucket]

    assign_popularity_bucket() using pd.qcut on popularity_score

    dataset.fit(users, items, user_features, item_features)
    Build: interactions, train_interactions, train_weights, test_interactions
    Build: user_features_matrix, item_features_matrix

    Return dict with all artifacts:
      {
        "dataset": dataset,
        "interactions": interactions,
        "train_interactions": train_interactions,
        "train_weights": train_weights,
        "test_interactions": test_interactions,
        "user_features_matrix": user_features_matrix,
        "item_features_matrix": item_features_matrix,
        "active_users": active_users,
        "active_items": active_items,
        "ife": ife,
        "ufe": ufe,
        "train_events_df": train_events_df,
        "test_events_raw_df": test_events_raw_df,
        "test_user_game_df": test_user_game_df,
        "user_feature_vocab": user_feature_vocab,
        "item_feature_vocab": item_feature_vocab,
      }
```

Notes:
- This step should own the temporal split and train-only table rebuild internally.
- The flow should not precompute full-data feature tables unless there is a separate explicit need.
- Keep the notebook’s fallback behavior for sparse data so the pipeline stays executable.

### `steps/train.py` — Model Training

**Notebook cell:** 35

```
@task
def train_model(dataset_artifacts, settings) -> LightFM:
    model = LightFM(
        no_components=settings.NO_COMPONENTS,
        loss=settings.LOSS,
        learning_rate=settings.LEARNING_RATE,
        item_alpha=settings.ITEM_ALPHA,
        user_alpha=settings.USER_ALPHA,
        random_state=settings.SEED,
    )

    for epoch in range(settings.N_EPOCHS):
        model.fit_partial(
            interactions=dataset_artifacts["train_interactions"],
            sample_weight=dataset_artifacts["train_weights"],
            user_features=dataset_artifacts["user_features_matrix"],
            item_features=dataset_artifacts["item_features_matrix"],
            num_threads=settings.PREDICT_NUM_THREADS,
            epochs=1,
        )
        # Log checkpoint metrics every 5 epochs (print for now, MLflow later)

    return model
```

### `steps/evaluate.py` — Evaluation

**Notebook cell:** 37

```
@task
def evaluate_model(model, dataset_artifacts, settings) -> dict:
    Compute:
      - train_precision_at_k (k=5)
      - test_precision_at_k (k=5)
      - train_auc
      - test_auc
      - ndcg_at_5

    Print summary table
    Return metrics dict
```

Notes:
- Evaluation should follow the notebook formulae.
- Do not treat live-data metric parity as a hard acceptance gate. Exact metric comparison only makes sense against a frozen fixture dataset.

### `steps/export.py` — Artifact Export

**From fastapi_plan_v2.md section 2 and current notebook exporter**

```
@task
def export_artifacts(model, dataset_artifacts, games_df, settings) -> str:
    Export to settings.ARTIFACT_DIR:
      - model.joblib
      - dataset.joblib
      - user_features_matrix.joblib
      - item_features_matrix.joblib
      - interactions.joblib (full, not train-only)
      - user_id_map.json
      - item_id_map.json
      - game_metadata.json (gameName from games_df, gameType/provider from ife)
      - popularity_ranking.json (sorted desc by popularity_score)

    Return artifact_dir path
```

Important:
- Preserve the current serving contract exactly.
- Export `interactions`, not `train_interactions`, because the API uses full observed history to exclude already-played items.
- Keep `gameId` and `userId` exported as strings.

---

## 4. Prefect Flow (`pipeline/flow.py`)

```python
from prefect import flow
from pipeline.config import PipelineSettings
from pipeline.steps import ingest, enrich, align, build_dataset, train, evaluate, export


@flow(name="betblitz-recsys-training", log_prints=True)
def training_flow():
    settings = PipelineSettings()

    # Stage 1: Data ingestion
    transactions_df = ingest.load_transactions(settings)
    games_df = ingest.load_game_details(settings)
    users_df = ingest.load_users(settings)
    transactions_df = ingest.clean_and_merge(transactions_df, games_df, settings)

    # Stage 2: Enrichment
    transactions_df = enrich.enrich_sessions(transactions_df, settings)
    transactions_df = enrich.enrich_device(transactions_df, users_df, settings)

    # Stage 3: Schema alignment
    events_df = align.align_to_schema(transactions_df)

    # Stage 4: LightFM dataset construction (includes temporal split)
    dataset_artifacts = build_dataset.build_lightfm_dataset(events_df, games_df, settings)

    # Stage 5: Training
    model = train.train_model(dataset_artifacts, settings)

    # Stage 6: Evaluation
    metrics = evaluate.evaluate_model(model, dataset_artifacts, settings)

    # Stage 7: Export artifacts
    artifact_path = export.export_artifacts(model, dataset_artifacts, games_df, settings)

    return {
        "metrics": metrics,
        "artifact_path": artifact_path,
    }
```

Notes:
- The flow should return both metrics and the artifact path for easier orchestration/debugging.
- Use `log_prints=True` initially for operational visibility.

---

## 5. CLI Entry Point (`pipeline/run.py`)

```python
from pipeline.flow import training_flow


if __name__ == "__main__":
    training_flow()
```

Run with:

```bash
# From repo root
python -m pipeline.run
```

For the nightly Prefect schedule (2AM), register the flow with Prefect:

```python
training_flow.serve(name="betblitz-nightly-training", cron="0 2 * * *")
```

This is a one-liner addition when you're ready for scheduling. Not needed for v0.2.

---

## 6. Requirements (`pipeline/requirements.txt`)

```
prefect>=3.0.0
pymongo>=4.6.0
pandas>=2.0.0
numpy>=1.24.0
lightfm==1.17
scipy>=1.10.0
scikit-learn>=1.3.0
joblib>=1.3.0
pydantic-settings>=2.1.0
```

Separate from the FastAPI service's requirements. Shared dependencies (`lightfm`, `numpy`, `scipy`, `joblib`) must be version-compatible.

**CRITICAL:** `lightfm==1.17` must match the serving environment exactly. Serialized models are not portable across LightFM versions.

---

## 7. Tests

```
tests/pipeline/test_ingest.py:
  - test_extract_bet_part → correctly extracts sportName, categoryName, tournamentName
  - test_game_name_normalization → 998:baccarat → Baccarat
  - test_clean_and_merge_no_row_inflation → assert len before == len after

tests/pipeline/test_align.py:
  - test_map_game_type → 'sic bo' → live_dealer, 'aviator' → crash
  - test_map_provider → 'EvoSW' → Evolution, hex ObjectId → unknown
  - test_map_outcome → 'WIN' → net_positive, 'LOSE' → net_negative, 'LOSS' → net_negative
  - test_map_device_type → 'desktop' → desktop, 'Android' → mobile
  - test_align_to_schema_columns → output has all 17 SCHEMA_COLUMNS

tests/pipeline/test_features.py:
  - test_implicit_score_range → all scores between 0 and 1
  - test_engagement_intensity_non_negative → no negative values
  - test_user_features_shape → one row per unique user

tests/pipeline/test_build_dataset.py:
  - test_holdout_split_no_leakage → holdout pairs not in train set
  - test_token_vocab_complete → user and item tokens match expected format
  - test_interactions_nnz → nnz > 0

tests/pipeline/test_train.py:
  - test_model_trains_without_error → fit_partial completes
  - test_model_can_predict → predict returns array of correct length

tests/pipeline/test_artifact_contract.py:
  - test_exported_artifacts_loadable_by_model_service → load all 9 artifacts using
    ModelService.load_artifacts(), call recommend() for a known user_id from
    user_id_map.json, call recommend() for an unknown user_id, assert both return
    valid response structures

tests/pipeline/test_parity_fixtures.py:
  - optional fixture-based parity test only on frozen input data
  - compare stable invariants such as row counts, token vocab, matrix shapes,
    and selected metrics against expected snapshots
```

Testing principle:
- Use live-data smoke tests for operational validity.
- Use frozen fixtures for reproducibility checks.
- Do not make live MongoDB metric parity a hard CI condition.

---

## 8. What is OUT OF SCOPE for v0.2

- **MLflow tracking** — separate task. The pipeline returns a metrics dict that MLflow can consume later.
- **Prometheus/Grafana** — serving-layer monitoring, not training-pipeline scope.
- **Redis feature store write** — future scope; pipeline exports artifacts only.
- **Neo4j embedding pipeline** — separate scope entirely.
- **EDA / diagnostic plots** — remain in the notebook.
- **Data simulation / augmentation** — out of scope.
- **Hyperparameter tuning** — out of scope.
- **Serving-layer API changes** — out of scope. The FastAPI contract stays unchanged.

---

## 9. Implementation Order

1. `pipeline/config.py` — settings
2. `pipeline/steps/align.py` — map functions (most testable, no MongoDB needed)
3. `pipeline/steps/ingest.py` — data loading
4. `pipeline/steps/enrich.py` — session + device joins
5. `pipeline/steps/features.py` — feature engineering helpers
6. `pipeline/steps/build_dataset.py` — LightFM dataset construction
7. `pipeline/steps/train.py` — model training
8. `pipeline/steps/evaluate.py` — metrics
9. `pipeline/steps/export.py` — artifact export
10. `pipeline/flow.py` — wire everything together
11. `pipeline/run.py` — CLI entry point
12. `tests/pipeline/` — all tests
13. Verify end-to-end: run pipeline → export artifacts → load in FastAPI → validate `/health` and `/recommend`

---

## 10. Verification Checklist

After implementation, confirm:

- [ ] `python -m pipeline.run` executes end-to-end without error
- [ ] Artifacts appear in `betblitz-recsys-api/artifacts/` (all 9 files)
- [ ] Exported artifacts are loadable by the current FastAPI `ModelService`
- [ ] `/health` reports valid loaded dimensions after swapping artifacts
- [ ] `/recommend` returns valid recommendations for a known user
- [ ] `/recommend` returns valid popularity fallback recommendations for a cold-start user
- [ ] Full interactions matrix is exported and exclude-played behavior still works
- [ ] Each step is independently importable and testable
- [ ] `pytest tests/pipeline/` passes
- [ ] Optional frozen-fixture parity checks pass

Notes:
- For live MongoDB runs, do not require exact metric equality with notebook output.
- If you need strict parity validation, run both notebook and modular pipeline against the same frozen fixture data.

---

## 11. Notebook-to-Module Cell Mapping

| Module | Notebook Cells | Key Functions |
|---|---|---|
| `ingest.py` | 2, 3, 4, 5, 6, 8 | `extract_bet_part()`, field rename, game merge |
| `enrich.py` | 14, 15 | exact `merge_asof` session join, exact device join cascade |
| `align.py` | 22, 24 | `map_game_type()`, `map_provider()`, `map_outcome()`, `align_to_event_schema()` |
| `features.py` | 26, 27, 29, 30, 34 (partial) | `build_feature_tables_from_events()` |
| `build_dataset.py` | 34 (split + dataset) | temporal holdout, `user_tokens()`, `item_tokens()`, `assign_popularity_bucket()` |
| `train.py` | 35 | `LightFM()`, `fit_partial()` with `sample_weight` |
| `evaluate.py` | 37 | `precision_at_k()`, `auc_score()`, `ndcg_score()` |
| `export.py` | current notebook export script / serving contract | `joblib.dump()`, `json.dump()` |

---

## 12. Final Instruction to Implementer

When notebook logic and engineering neatness conflict, choose notebook parity for v0.2 unless a change is explicitly marked as an intentional divergence.

This pipeline is successful when it does the following reliably:
- trains from the same data source,
- exports the same serving artifacts,
- loads cleanly in the existing FastAPI backend,
- and returns ranked game recommendations to the backend without changing the serving contract.