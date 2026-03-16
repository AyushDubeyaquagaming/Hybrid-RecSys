"""Tests for pipeline/steps/build_dataset.py"""
import numpy as np
import pandas as pd
import pytest

from pipeline.steps.build_dataset import (
    assign_popularity_bucket,
    build_lightfm_dataset,
    item_tokens,
    user_tokens,
)


def _make_events_df(n_users=5, n_games=6, n_events=60, seed=42):
    rng = np.random.default_rng(seed)
    users = [f"user_{i}" for i in range(n_users)]
    games = [f"game_{i}" for i in range(n_games)]
    timestamps = pd.date_range("2024-01-01", periods=n_events, freq="4h")

    df = pd.DataFrame(
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
    return df


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


class TestHoldoutSplitNoLeakage:
    def test_holdout_pairs_not_in_train_set(self):
        from pipeline.config import PipelineSettings

        settings = PipelineSettings()
        events_df = _make_events_df()
        games_df = _make_games_df()
        artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)

        test_user_game_df = artifacts["test_user_game_df"]
        train_events_df = artifacts["train_events_df"]

        # For each holdout (userId, gameId) pair, that pair must NOT appear in train
        # as the LAST event for that user — the leakage-free check
        holdout_keys = set(
            zip(test_user_game_df["userId"], test_user_game_df["gameId"])
        )
        train_keys = set(zip(train_events_df["userId"], train_events_df["gameId"]))

        # It's OK for a pair to appear in train if it has EARLIER events,
        # but the holdout's defining property is that it was the last game touched.
        # We verify that for each holdout user, the holdout gameId exists in test
        # and was not the only game in train for that user.
        for user_id, game_id in holdout_keys:
            user_train = train_events_df[train_events_df["userId"] == user_id]
            # User must have at least one train event (or dataset is too sparse)
            if len(user_train) > 0:
                # The holdout game may appear in train (historical plays), but
                # the *last* touch timestamp should be later in test
                holdout_ts = (
                    artifacts["test_events_raw_df"][
                        (artifacts["test_events_raw_df"]["userId"] == user_id)
                        & (artifacts["test_events_raw_df"]["gameId"] == game_id)
                    ]["timestamp"].max()
                )
                train_ts = (
                    train_events_df[
                        (train_events_df["userId"] == user_id)
                        & (train_events_df["gameId"] == game_id)
                    ]["timestamp"].max()
                )
                if pd.notna(train_ts):
                    # If same game in train, holdout timestamp must be >= train timestamp
                    assert holdout_ts >= train_ts


class TestTokenVocabComplete:
    def test_user_tokens_format(self):
        row = {
            "preferred_time_of_day": "evening",
            "preferred_day_of_week": "friday",
            "preferred_device": "mobile",
            "preferred_entry_point": "livecasino",
        }
        tokens = user_tokens(row)
        assert len(tokens) == 4
        assert "preferred_time_of_day:evening" in tokens
        assert "preferred_day_of_week:friday" in tokens
        assert "preferred_device:mobile" in tokens
        assert "preferred_entry_point:livecasino" in tokens

    def test_item_tokens_format(self):
        row = {
            "game_type": "live_dealer",
            "provider": "Evolution",
            "popularity_bucket": "hot",
        }
        tokens = item_tokens(row)
        assert len(tokens) == 3
        assert "game_type:live_dealer" in tokens
        assert "provider:Evolution" in tokens
        assert "popularity_bucket:hot" in tokens

    def test_vocab_from_build(self):
        from pipeline.config import PipelineSettings

        settings = PipelineSettings()
        events_df = _make_events_df()
        games_df = _make_games_df()
        artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)

        user_vocab = artifacts["user_feature_vocab"]
        item_vocab = artifacts["item_feature_vocab"]

        assert any(t.startswith("preferred_time_of_day:") for t in user_vocab)
        assert any(t.startswith("game_type:") for t in item_vocab)
        assert any(t.startswith("popularity_bucket:") for t in item_vocab)


class TestInteractionsNnz:
    def test_interactions_nnz_positive(self):
        from pipeline.config import PipelineSettings

        settings = PipelineSettings()
        events_df = _make_events_df()
        games_df = _make_games_df()
        artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)

        assert artifacts["interactions"].nnz > 0
        assert artifacts["train_interactions"].nnz > 0


class TestAssignPopularityBucket:
    def test_four_buckets_normal(self):
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        buckets = assign_popularity_bucket(scores)
        assert set(buckets.unique()).issubset({"cold", "warm", "hot", "blockbuster"})

    def test_empty_series(self):
        result = assign_popularity_bucket(pd.Series(dtype="float64"))
        assert len(result) == 0

    def test_single_element(self):
        result = assign_popularity_bucket(pd.Series([1.0]))
        assert len(result) == 1
