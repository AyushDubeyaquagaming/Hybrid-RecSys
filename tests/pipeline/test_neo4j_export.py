"""Tests for pipeline/steps/neo4j_export.py

Unit tests only — no live Neo4j instance required.
Tests cover: embedding extraction, row preparation, batching, and the
disabled/best-effort guard. Integration verification happens via make retrain.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineSettings
from pipeline.steps.build_dataset import build_lightfm_dataset
from pipeline.steps.neo4j_export import (
    _extract_embeddings,
    _iter_batches,
    _prepare_game_rows,
    _prepare_player_rows,
    export_embeddings_to_neo4j,
)
from pipeline.steps.train import train_model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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
def trained_artifacts():
    """Build dataset and train a minimal model once for all tests in this module."""
    settings = PipelineSettings(N_EPOCHS=2)
    events_df = _make_events_df()
    games_df = _make_games_df()
    dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
    train_result = train_model.fn(dataset_artifacts, settings)
    return train_result["model"], dataset_artifacts, games_df


# ---------------------------------------------------------------------------
# TestExtractEmbeddings
# ---------------------------------------------------------------------------

class TestExtractEmbeddings:
    def test_returns_correct_user_embedding_shape(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        n_users = len(dataset_artifacts["active_users"])
        assert emb_data["user_embeddings"].shape == (n_users, model.no_components)

    def test_returns_correct_item_embedding_shape(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        n_items = len(dataset_artifacts["active_items"])
        assert emb_data["item_embeddings"].shape == (n_items, model.no_components)

    def test_user_biases_are_1d_float_array(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        biases = emb_data["user_biases"]
        assert biases.ndim == 1
        assert biases.dtype in (np.float32, np.float64)

    def test_item_biases_are_1d_float_array(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        biases = emb_data["item_biases"]
        assert biases.ndim == 1
        assert biases.dtype in (np.float32, np.float64)

    def test_active_users_match_dataset_artifacts(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        assert set(emb_data["active_users"]) == set(dataset_artifacts["active_users"])

    def test_active_items_match_dataset_artifacts(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        assert set(emb_data["active_items"]) == set(dataset_artifacts["active_items"])


# ---------------------------------------------------------------------------
# TestEmbeddingDataPreparation
# ---------------------------------------------------------------------------

class TestEmbeddingDataPreparation:
    def test_player_rows_have_required_fields(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        rows = _prepare_player_rows(emb_data, player_key="id")
        assert len(rows) == len(dataset_artifacts["active_users"])
        for row in rows:
            assert "id" in row
            assert "embedding" in row
            assert "bias" in row

    def test_player_rows_use_custom_player_key(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        rows = _prepare_player_rows(emb_data, player_key="player_id")
        for row in rows:
            assert "player_id" in row
            assert "id" not in row

    def test_game_rows_have_required_fields(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        rows = _prepare_game_rows(emb_data, game_key="id")
        assert len(rows) == len(dataset_artifacts["active_items"])
        for row in rows:
            assert "id" in row
            assert "embedding" in row
            assert "bias" in row

    def test_game_rows_use_custom_game_key(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        rows = _prepare_game_rows(emb_data, game_key="game_id")
        for row in rows:
            assert "game_id" in row
            assert "id" not in row

    def test_embedding_is_list_of_floats(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        player_rows = _prepare_player_rows(emb_data, player_key="id")
        game_rows = _prepare_game_rows(emb_data, game_key="id")
        for row in player_rows + game_rows:
            assert isinstance(row["embedding"], list)
            assert all(isinstance(v, float) for v in row["embedding"])

    def test_embedding_length_matches_no_components(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        player_rows = _prepare_player_rows(emb_data, player_key="id")
        game_rows = _prepare_game_rows(emb_data, game_key="id")
        for row in player_rows + game_rows:
            assert len(row["embedding"]) == model.no_components

    def test_bias_is_python_float(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        player_rows = _prepare_player_rows(emb_data, player_key="id")
        for row in player_rows:
            assert isinstance(row["bias"], float)

    def test_player_ids_are_strings(self, trained_artifacts):
        model, dataset_artifacts, _ = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        rows = _prepare_player_rows(emb_data, player_key="id")
        for row in rows:
            assert isinstance(row["id"], str)

    def test_game_ids_are_strings(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        emb_data = _extract_embeddings(model, dataset_artifacts)
        rows = _prepare_game_rows(emb_data, game_key="id")
        for row in rows:
            assert isinstance(row["id"], str)


# ---------------------------------------------------------------------------
# TestBatching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_splits_by_batch_size(self):
        rows = list(range(10))
        batches = list(_iter_batches(rows, batch_size=3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[3] == [9]

    def test_single_batch_when_rows_lt_batch_size(self):
        rows = list(range(5))
        batches = list(_iter_batches(rows, batch_size=100))
        assert len(batches) == 1
        assert batches[0] == rows

    def test_empty_rows_produces_no_batches(self):
        batches = list(_iter_batches([], batch_size=10))
        assert batches == []

    def test_exact_multiple_of_batch_size(self):
        rows = list(range(9))
        batches = list(_iter_batches(rows, batch_size=3))
        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)


# ---------------------------------------------------------------------------
# TestNeo4jDisabled
# ---------------------------------------------------------------------------

class TestNeo4jDisabled:
    def test_returns_zero_when_disabled(self, trained_artifacts):
        """export_embeddings_to_neo4j returns 0 immediately with NEO4J_ENABLED=False."""
        model, dataset_artifacts, games_df = trained_artifacts
        settings = PipelineSettings(NEO4J_ENABLED=False)
        result = export_embeddings_to_neo4j.fn(
            model, dataset_artifacts, settings
        )
        assert result == 0


class TestNeo4jSuccessfulWritePath:
    def test_returns_actual_matched_update_counts(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        settings = PipelineSettings(
            NEO4J_ENABLED=True,
            NEO4J_PLAYER_KEY="player_id",
            NEO4J_GAME_KEY="game_id",
            NEO4J_BATCH_SIZE=2,
        )

        driver = MagicMock()

        with patch("pipeline.steps.neo4j_export._get_neo4j_driver", return_value=driver):
            with patch("pipeline.steps.neo4j_export._ensure_constraints"):
                with patch(
                    "pipeline.steps.neo4j_export._write_player_embeddings",
                    side_effect=[2, 2, 1],
                ) as write_players:
                    with patch(
                        "pipeline.steps.neo4j_export._write_game_embeddings",
                        side_effect=[2, 2, 2],
                    ) as write_games:
                        result = export_embeddings_to_neo4j.fn(
                            model,
                            dataset_artifacts,
                            settings,
                            player_key=settings.NEO4J_PLAYER_KEY,
                            game_key=settings.NEO4J_GAME_KEY,
                        )

        assert result == 11
        assert write_players.call_count == 3
        assert write_games.call_count == 3
        assert write_players.call_args_list[0].args[3] == "player_id"
        assert write_games.call_args_list[0].args[3] == "game_id"
        driver.verify_connectivity.assert_called_once()
        driver.close.assert_called_once()

    def test_best_effort_returns_zero_when_write_fails_after_connectivity(self, trained_artifacts):
        model, dataset_artifacts, games_df = trained_artifacts
        settings = PipelineSettings(NEO4J_ENABLED=True)

        driver = MagicMock()

        with patch("pipeline.steps.neo4j_export._get_neo4j_driver", return_value=driver):
            with patch("pipeline.steps.neo4j_export._ensure_constraints"):
                with patch(
                    "pipeline.steps.neo4j_export._write_player_embeddings",
                    side_effect=RuntimeError("write failed"),
                ):
                    result = export_embeddings_to_neo4j.fn(
                        model,
                        dataset_artifacts,
                        settings,
                    )

        assert result == 0
        driver.close.assert_called_once()

    def test_best_effort_on_unreachable_host(self, trained_artifacts):
        """export_embeddings_to_neo4j returns 0 gracefully when Neo4j is unreachable."""
        model, dataset_artifacts, games_df = trained_artifacts
        settings = PipelineSettings(
            NEO4J_ENABLED=True,
            NEO4J_URI="bolt://127.0.0.1:19999",  # port nobody is listening on
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="invalid",
        )
        result = export_embeddings_to_neo4j.fn(
            model, dataset_artifacts, settings
        )
        assert result == 0
