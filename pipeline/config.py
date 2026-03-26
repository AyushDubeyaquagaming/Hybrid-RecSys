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

    # Logging
    LOG_LEVEL: str = "INFO"

    # MLflow
    MLFLOW_ENABLED: bool = False
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "betblitz-recsys"

    # MLflow Model Registry
    MLFLOW_REGISTRY_ENABLED: bool = True
    MLFLOW_REGISTERED_MODEL_NAME: str = "betblitz-lightfm-recsys"

    # MLflow diagnostics / nightly EDA
    EDA_ENABLED: bool = True
    EDA_MAX_GAMES_TO_PLOT: int = 15

    # Redis Feature Store
    REDIS_ENABLED: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_TTL_SECONDS: int = 90000  # ~25 hours

    # Neo4j Embedding Store
    NEO4J_ENABLED: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_PLAYER_KEY: str = "id"
    NEO4J_GAME_KEY: str = "id"
    NEO4J_BATCH_SIZE: int = 500

    # Output
    ARTIFACT_DIR: str = str(
        Path(__file__).resolve().parent.parent / "betblitz-recsys-api" / "artifacts"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
