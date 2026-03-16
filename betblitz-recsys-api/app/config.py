from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "betblitz-recsys-api"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Model artifacts path
    ARTIFACT_DIR: str = "./artifacts"

    # Recommendation defaults
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20

    # LightFM predict threads (use 1 in serving — FastAPI handles concurrency)
    PREDICT_NUM_THREADS: int = 1

    # Redis Feature Store (informational health check only in v0.1)
    REDIS_ENABLED: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
