"""Tests for pipeline/steps/train.py"""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineSettings
from pipeline.steps.build_dataset import build_lightfm_dataset
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


@pytest.fixture(scope="module")
def dataset_artifacts():
    settings = PipelineSettings()
    settings_fast = PipelineSettings(N_EPOCHS=2)  # fast for tests
    events_df = _make_events_df()
    games_df = _make_games_df()
    return build_lightfm_dataset.fn(events_df, games_df, settings)


class TestModelTrainsWithoutError:
    def test_fit_partial_completes(self, dataset_artifacts):
        settings = PipelineSettings(N_EPOCHS=2)
        result = train_model.fn(dataset_artifacts, settings)
        assert result["model"] is not None

    def test_model_has_components(self, dataset_artifacts):
        settings = PipelineSettings(N_EPOCHS=1)
        model = train_model.fn(dataset_artifacts, settings)["model"]
        assert hasattr(model, "item_embeddings")
        assert model.item_embeddings is not None


class TestModelCanPredict:
    def test_predict_returns_correct_length(self, dataset_artifacts):
        settings = PipelineSettings(N_EPOCHS=2)
        model = train_model.fn(dataset_artifacts, settings)["model"]

        n_items = dataset_artifacts["interactions"].shape[1]
        scores = model.predict(
            user_ids=np.repeat(0, n_items),
            item_ids=np.arange(n_items),
            user_features=dataset_artifacts["user_features_matrix"],
            item_features=dataset_artifacts["item_features_matrix"],
            num_threads=1,
        )
        assert len(scores) == n_items

    def test_predict_returns_finite_scores(self, dataset_artifacts):
        settings = PipelineSettings(N_EPOCHS=2)
        model = train_model.fn(dataset_artifacts, settings)["model"]

        n_items = dataset_artifacts["interactions"].shape[1]
        scores = model.predict(
            user_ids=np.repeat(0, n_items),
            item_ids=np.arange(n_items),
            user_features=dataset_artifacts["user_features_matrix"],
            item_features=dataset_artifacts["item_features_matrix"],
            num_threads=1,
        )
        assert np.isfinite(scores).all()
