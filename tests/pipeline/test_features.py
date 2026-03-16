"""Tests for pipeline/steps/features.py"""
import numpy as np
import pandas as pd
import pytest

from pipeline.steps.features import build_feature_tables_from_events


def _make_events_df(n_users=3, n_games=4, n_events=20, seed=42):
    """Generate a synthetic events DataFrame matching SCHEMA_COLUMNS."""
    rng = np.random.default_rng(seed)

    users = [f"user_{i}" for i in range(n_users)]
    games = [f"game_{i}" for i in range(n_games)]
    sessions = [f"sess_{i}" for i in range(n_events)]

    timestamps = pd.date_range("2024-01-01", periods=n_events, freq="6h")

    df = pd.DataFrame(
        {
            "eventType": "game_session",
            "userId": rng.choice(users, n_events),
            "sessionId": sessions,
            "gameId": rng.choice(games, n_events),
            "gameType": rng.choice(
                ["live_dealer", "slot", "crash"], n_events
            ),
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
            "entryPoint": rng.choice(
                ["livecasino", "direct", "casino"], n_events
            ),
        }
    )
    return df


class TestImplicitScoreRange:
    def test_scores_between_0_and_1(self):
        events_df = _make_events_df()
        _, _, _, _, user_game_df = build_feature_tables_from_events(events_df)
        assert (user_game_df["implicit_score"] >= 0).all()
        assert (user_game_df["implicit_score"] <= 1).all()


class TestEngagementIntensityNonNegative:
    def test_engagement_intensity_non_negative(self):
        events_df = _make_events_df()
        fe_events, _, _, _, _ = build_feature_tables_from_events(events_df)
        assert (fe_events["engagement_intensity"] >= 0).all()


class TestUserFeaturesShape:
    def test_one_row_per_unique_user(self):
        events_df = _make_events_df(n_users=5, n_events=50)
        _, user_features_df, _, _, _ = build_feature_tables_from_events(events_df)
        n_users_in_events = events_df["userId"].nunique()
        assert len(user_features_df) == n_users_in_events

    def test_user_id_is_index(self):
        events_df = _make_events_df()
        _, user_features_df, _, _, _ = build_feature_tables_from_events(events_df)
        assert "userId" in user_features_df.columns

    def test_preferred_time_of_day_present(self):
        events_df = _make_events_df()
        _, user_features_df, _, _, _ = build_feature_tables_from_events(events_df)
        assert "preferred_time_of_day" in user_features_df.columns

    def test_game_features_one_row_per_game(self):
        events_df = _make_events_df(n_games=4, n_events=40)
        _, _, game_features_df, _, _ = build_feature_tables_from_events(events_df)
        n_games_in_events = events_df["gameId"].nunique()
        assert len(game_features_df) == n_games_in_events

    def test_popularity_score_non_negative(self):
        events_df = _make_events_df()
        _, _, game_features_df, _, _ = build_feature_tables_from_events(events_df)
        assert (game_features_df["popularity_score"] >= 0).all()
