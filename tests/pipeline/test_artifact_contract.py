"""Tests verifying exported artifacts are loadable by the FastAPI ModelService.

These tests build and export artifacts using the pipeline, then load them via
ModelService and verify that recommend() returns valid response structures.
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineSettings
from pipeline.steps.build_dataset import build_lightfm_dataset
from pipeline.steps.export import export_artifacts
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
def exported_artifact_dir():
    """Build, train, and export artifacts to a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = PipelineSettings(ARTIFACT_DIR=tmpdir, N_EPOCHS=2)
        events_df = _make_events_df()
        games_df = _make_games_df()

        dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
        model = train_model.fn(dataset_artifacts, settings)["model"]
        export_artifacts.fn(model, dataset_artifacts, games_df, settings)
        yield tmpdir


class TestExportedArtifactsLoadableByModelService:
    REQUIRED_FILES = [
        "model.joblib",
        "dataset.joblib",
        "user_features_matrix.joblib",
        "item_features_matrix.joblib",
        "interactions.joblib",
        "user_id_map.json",
        "item_id_map.json",
        "game_metadata.json",
        "popularity_ranking.json",
    ]

    def test_all_artifact_files_exist(self, exported_artifact_dir):
        for fname in self.REQUIRED_FILES:
            assert os.path.exists(
                os.path.join(exported_artifact_dir, fname)
            ), f"Missing artifact: {fname}"

    def test_artifacts_loadable_by_model_service(self, exported_artifact_dir):
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../betblitz-recsys-api")
        )
        from app.services.model_service import ModelService

        svc = ModelService()
        svc.load_artifacts(exported_artifact_dir, num_threads=1)
        assert svc.is_loaded()

    def test_recommend_known_user_returns_valid_structure(self, exported_artifact_dir):
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../betblitz-recsys-api")
        )
        from app.services.model_service import ModelService

        with open(os.path.join(exported_artifact_dir, "user_id_map.json")) as f:
            user_id_map = json.load(f)

        known_user_id = next(iter(user_id_map.keys()))

        svc = ModelService()
        svc.load_artifacts(exported_artifact_dir, num_threads=1)
        result = svc.recommend(user_id=known_user_id, top_k=5, exclude_played=True)

        assert "recommendations" in result
        assert "is_cold_start" in result
        assert "source" in result
        assert result["is_cold_start"] is False
        assert result["source"] == "lightfm"
        assert isinstance(result["recommendations"], list)

        for rec in result["recommendations"]:
            assert "game_id" in rec
            assert "game_name" in rec
            assert "score" in rec
            assert "rank" in rec

    def test_recommend_unknown_user_returns_popularity_fallback(
        self, exported_artifact_dir
    ):
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../betblitz-recsys-api")
        )
        from app.services.model_service import ModelService

        svc = ModelService()
        svc.load_artifacts(exported_artifact_dir, num_threads=1)
        result = svc.recommend(
            user_id="totally_unknown_user_xyz", top_k=5, exclude_played=True
        )

        assert result["is_cold_start"] is True
        assert result["source"] == "popularity_fallback"
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0
