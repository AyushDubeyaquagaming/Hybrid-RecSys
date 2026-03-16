import numpy as np
import pandas as pd
from prefect import task

from pipeline.exceptions import DataValidationError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


def _mode_or_default(series, default_value):
    mode = series.mode()
    return mode.iat[0] if len(mode) else default_value


def build_feature_tables_from_events(base_events: pd.DataFrame) -> tuple:
    """Build user, game, provider, and user-game feature tables from events.

    Returns:
        (fe_events, user_features_df, game_features_df, provider_features_df, user_game_df)
    """
    if base_events.empty:
        raise DataValidationError("Cannot build feature tables from an empty events dataframe.")
    fe_local = (
        base_events.copy().sort_values(["userId", "timestamp"]).reset_index(drop=True)
    )
    fe_local["is_quick_exit"] = (fe_local["exitType"] == "quick_exit").astype(int)
    fe_local["is_positive_outcome"] = (fe_local["outcome"] == "net_positive").astype(int)

    stake_weight_local = (
        fe_local["stakeLevelCategory"].map({"low": 1, "medium": 2, "high": 3}).fillna(1)
    )
    fe_local["engagement_intensity"] = (
        np.log1p(fe_local["durationSeconds"].clip(lower=0)) * 0.5
        + fe_local["roundsPlayed"].clip(lower=0) * 0.3
        + stake_weight_local * 0.2
    )

    user_features_local = (
        fe_local.groupby("userId")
        .agg(
            total_sessions=("sessionId", "count"),
            unique_games=("gameId", "nunique"),
            unique_providers=("provider", "nunique"),
            unique_game_types=("gameType", "nunique"),
            avg_duration_sec=("durationSeconds", "mean"),
            avg_rounds=("roundsPlayed", "mean"),
            quick_exit_rate=("is_quick_exit", "mean"),
            return_10m_rate=("returnedWithin10mins", "mean"),
            positive_outcome_rate=("is_positive_outcome", "mean"),
            avg_engagement_intensity=("engagement_intensity", "mean"),
            preferred_time_of_day=(
                "timeOfDay",
                lambda x: _mode_or_default(x, "late_night"),
            ),
            preferred_day_of_week=(
                "dayOfWeek",
                lambda x: _mode_or_default(x, "friday"),
            ),
            preferred_device=("deviceType", lambda x: _mode_or_default(x, "unknown")),
            preferred_entry_point=(
                "entryPoint",
                lambda x: _mode_or_default(x, "unknown"),
            ),
            last_event_ts=("timestamp", "max"),
        )
        .reset_index()
    )

    max_ts_local = fe_local["timestamp"].max()
    user_features_local["recency_days"] = (
        max_ts_local - user_features_local["last_event_ts"]
    ).dt.total_seconds().div(86400)

    game_features_local = (
        fe_local.groupby("gameId")
        .agg(
            game_sessions=("sessionId", "count"),
            unique_users=("userId", "nunique"),
            game_type=("gameType", lambda x: _mode_or_default(x, "live_dealer")),
            provider=("provider", lambda x: _mode_or_default(x, "unknown")),
            avg_duration_sec=("durationSeconds", "mean"),
            avg_rounds=("roundsPlayed", "mean"),
            quick_exit_rate=("is_quick_exit", "mean"),
            return_10m_rate=("returnedWithin10mins", "mean"),
            positive_outcome_rate=("is_positive_outcome", "mean"),
        )
        .reset_index()
    )
    game_features_local["popularity_score"] = np.log1p(
        game_features_local["game_sessions"]
    )

    provider_features_local = (
        fe_local.groupby("provider")
        .agg(
            provider_sessions=("sessionId", "count"),
            provider_unique_users=("userId", "nunique"),
            provider_avg_engagement=("engagement_intensity", "mean"),
        )
        .reset_index()
    )

    user_game_local = (
        fe_local.groupby(["userId", "gameId"])
        .agg(
            interaction_count=("sessionId", "count"),
            avg_duration_sec=("durationSeconds", "mean"),
            avg_rounds=("roundsPlayed", "mean"),
            positive_outcome_rate=("is_positive_outcome", "mean"),
            return_10m_rate=("returnedWithin10mins", "mean"),
            avg_engagement_intensity=("engagement_intensity", "mean"),
            last_interaction_ts=("timestamp", "max"),
            dominant_game_type=(
                "gameType",
                lambda x: _mode_or_default(x, "live_dealer"),
            ),
            dominant_provider=("provider", lambda x: _mode_or_default(x, "unknown")),
        )
        .reset_index()
    )

    user_game_local["recency_days"] = (
        max_ts_local - user_game_local["last_interaction_ts"]
    ).dt.total_seconds().div(86400)

    freq_score_local = np.minimum(user_game_local["interaction_count"] / 20.0, 1.0)
    engagement_denom = max(
        user_game_local["avg_engagement_intensity"].quantile(0.95), 1e-6
    )
    engagement_score_local = np.minimum(
        user_game_local["avg_engagement_intensity"] / engagement_denom, 1.0
    )
    quality_score_local = user_game_local["positive_outcome_rate"].fillna(0)
    recency_weight_local = np.where(
        user_game_local["recency_days"] <= 7,
        1.0,
        np.where(
            user_game_local["recency_days"] <= 30,
            0.7,
            np.where(user_game_local["recency_days"] <= 90, 0.4, 0.2),
        ),
    )
    user_game_local["implicit_score"] = (
        0.40 * freq_score_local
        + 0.25 * engagement_score_local
        + 0.15 * quality_score_local
        + 0.20 * recency_weight_local
    ).round(4)

    return (
        fe_local,
        user_features_local,
        game_features_local,
        provider_features_local,
        user_game_local,
    )


@task
def build_feature_tables(events_df: pd.DataFrame) -> tuple:
    """Thin task wrapper around build_feature_tables_from_events."""
    logger.info("Building feature tables from %s events", len(events_df))
    result = build_feature_tables_from_events(events_df)
    fe_events, user_features_df, game_features_df, provider_features_df, user_game_df = result
    print(f"User features shape: {user_features_df.shape}")
    print(f"Game features shape: {game_features_df.shape}")
    print(f"Provider features shape: {provider_features_df.shape}")
    print(f"User-game interactions: {user_game_df.shape}")
    print(user_game_df["implicit_score"].describe().round(3))
    return result
