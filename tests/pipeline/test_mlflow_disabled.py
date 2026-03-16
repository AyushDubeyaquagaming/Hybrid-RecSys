"""Test that the training flow runs correctly with MLflow disabled."""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineSettings
from pipeline.steps.build_dataset import build_lightfm_dataset
from pipeline.steps.evaluate import evaluate_model
from pipeline.steps.export import export_artifacts, write_features_to_redis
from pipeline.steps.train import train_model


def _make_events_df(n_users=5, n_games=6, n_events=60, seed=42):
    rng = np.random.default_rng(seed)
    users = [f"user_{i}" for i in range(n_users)]
    games = [f"game_{i}" for i in range(n_games)]
    timestamps = pd.date_range("2024-01-01", periods=n_events, freq="4h")
    return pd.DataFrame(
        {
            "eventType": "game_session",
            "userId": rng.choice(users, n_events),
            "sessionId": [f"sess_{i}" for i in range(n_events)],
            "gameId": rng.choice(games, n_events),
            "gameType": rng.choice(["live_dealer", "slot", "crash"], n_events),
            "provider": rng.choice(["Evolution", "EZUGI", "Spribe"], n_events),
            "timestamp": timestamps,
            "durationSeconds": rng.uniform(0, 600, n_events),
            "roundsPlayed": rng.integers(1, 10, n_events),
            "stakeLevelCategory": rng.choice(["low", "medium", "high"], n_events),
            "outcome": rng.choice(
                ["net_positive", "net_negative", "break_even"], n_events
            ),
            "exitType": rng.choice(
                ["quick_exit", "natural_end", "returned_quickly", "unknown"], n_events
            ),
            "returnedWithin10mins": rng.choice([True, False], n_events),
            "deviceType": rng.choice(["mobile", "desktop", "unknown"], n_events),
            "timeOfDay": rng.choice(
                ["morning", "afternoon", "evening", "late_night"], n_events
            ),
            "dayOfWeek": rng.choice(
                ["monday", "tuesday", "wednesday", "thursday", "friday"], n_events
            ),
            "entryPoint": rng.choice(["livecasino", "direct", "casino"], n_events),
        }
    )


def _make_games_df():
    return pd.DataFrame(
        {
            "gameId": [f"game_{i}" for i in range(6)],
            "gameName": [f"Game {i}" for i in range(6)],
            "gameType": ["Live Casino"] * 6,
            "gamevendor": ["Evolution"] * 6,
            "minBet": [1.0] * 6,
            "maxBet": [100.0] * 6,
        }
    )


def test_pipeline_runs_with_mlflow_disabled(tmp_path):
    """Full pipeline (build → train → evaluate → export) completes with MLFLOW_ENABLED=false."""
    settings = PipelineSettings(
        MLFLOW_ENABLED=False,
        REDIS_ENABLED=False,
        ARTIFACT_DIR=str(tmp_path),
        N_EPOCHS=2,
    )
    events_df = _make_events_df()
    games_df = _make_games_df()

    dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
    model = train_model.fn(dataset_artifacts, settings)
    metrics = evaluate_model.fn(model, dataset_artifacts, settings)
    artifact_path = export_artifacts.fn(model, dataset_artifacts, games_df, settings)

    assert artifact_path == str(tmp_path)
    assert "train_precision_at_k" in metrics
    assert "test_precision_at_k" in metrics


def test_redis_write_skipped_when_disabled(tmp_path):
    """write_features_to_redis returns 0 immediately when REDIS_ENABLED=false."""
    settings = PipelineSettings(REDIS_ENABLED=False, ARTIFACT_DIR=str(tmp_path), N_EPOCHS=1)
    events_df = _make_events_df()
    games_df = _make_games_df()

    dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
    result = write_features_to_redis.fn(dataset_artifacts, settings)
    assert result == 0


def test_redis_write_best_effort_on_unreachable_host(tmp_path):
    """write_features_to_redis returns 0 gracefully when Redis is unreachable."""
    settings = PipelineSettings(
        REDIS_ENABLED=True,
        REDIS_HOST="127.0.0.1",
        REDIS_PORT=19999,  # port nobody is listening on
        ARTIFACT_DIR=str(tmp_path),
        N_EPOCHS=1,
    )
    events_df = _make_events_df()
    games_df = _make_games_df()

    dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
    result = write_features_to_redis.fn(dataset_artifacts, settings)
    assert result == 0
