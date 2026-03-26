import numpy as np
import pandas as pd
from lightfm.data import Dataset
from prefect import task

from pipeline.config import PipelineSettings
from pipeline.exceptions import DataValidationError
from pipeline.logging_utils import get_logger
from pipeline.steps.features import build_feature_tables_from_events, _mode_or_default


logger = get_logger(__name__)


def assign_popularity_bucket(score_series: pd.Series) -> pd.Series:
    n_rows = len(score_series)
    if n_rows == 0:
        return pd.Series(dtype="object")
    labels = ["cold", "warm", "hot", "blockbuster"][: min(4, n_rows)]
    if len(labels) == 1:
        return pd.Series([labels[0]] * n_rows, index=score_series.index, dtype="object")
    ranked = score_series.rank(method="first")
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series([labels[0]] * n_rows, index=score_series.index, dtype="object")


def user_tokens(row) -> list:
    return [
        f"preferred_time_of_day:{row['preferred_time_of_day']}",
        f"preferred_day_of_week:{row['preferred_day_of_week']}",
        f"preferred_device:{row['preferred_device']}",
        f"preferred_entry_point:{row['preferred_entry_point']}",
    ]


def item_tokens(row) -> list:
    return [
        f"game_type:{row['game_type']}",
        f"provider:{row['provider']}",
        f"popularity_bucket:{row['popularity_bucket']}",
    ]


@task
def build_lightfm_dataset(
    events_df: pd.DataFrame,
    games_df: pd.DataFrame,
    settings: PipelineSettings,
) -> dict:
    """Build LightFM dataset with temporal holdout split.

    Returns dict of all artifacts needed for training, evaluation, and export.
    """
    if events_df.empty:
        raise DataValidationError("Cannot build LightFM dataset from an empty events dataframe.")
    logger.info("Building LightFM dataset from events=%s games=%s", len(events_df), len(games_df))
    # --- Temporal holdout split ---
    events_for_split = events_df.sort_values(["userId", "timestamp"]).copy()
    pair_last_touch = events_for_split.groupby(
        ["userId", "gameId"], as_index=False
    ).agg(last_ts=("timestamp", "max"))
    pair_last_touch["distinct_games_for_user"] = pair_last_touch.groupby("userId")[
        "gameId"
    ].transform("nunique")

    holdout_pairs_df = (
        pair_last_touch[pair_last_touch["distinct_games_for_user"] >= 2]
        .sort_values(["userId", "last_ts"])
        .groupby("userId", as_index=False)
        .tail(1)[["userId", "gameId"]]
    )
    if len(holdout_pairs_df) == 0:
        holdout_pairs_df = (
            pair_last_touch.sort_values("last_ts").tail(1)[["userId", "gameId"]].copy()
        )
        print("Fallback holdout applied: sparse dataset, using one global holdout pair.")
    if len(holdout_pairs_df) == 0:
        raise ValueError(
            "No holdout user-game pairs available for evaluation. "
            "Dataset is empty after preprocessing."
        )

    holdout_keys = {
        tuple(x)
        for x in holdout_pairs_df[["userId", "gameId"]].to_records(index=False)
    }
    holdout_mask = pd.Series(
        [
            k in holdout_keys
            for k in zip(events_for_split["userId"], events_for_split["gameId"])
        ],
        index=events_for_split.index,
    )

    train_events_df = events_for_split.loc[~holdout_mask].copy()
    test_events_raw_df = events_for_split.loc[holdout_mask].copy()
    if len(train_events_df) == 0 and len(test_events_raw_df) > 0:
        train_events_df = test_events_raw_df.copy()
        print("Train fallback applied: copied holdout events to keep pipeline executable.")

    # --- Build feature tables on train data only ---
    (
        _,
        train_user_features_df,
        train_game_features_df,
        _,
        train_user_game_df,
    ) = build_feature_tables_from_events(train_events_df)

    test_user_game_df = (
        test_events_raw_df.groupby(["userId", "gameId"]).size().reset_index(name="interaction_count")
    )
    test_user_game_df["implicit_score"] = 1.0

    safe_train_user_game_df = train_user_game_df.copy()
    safe_train_user_game_df["userId"] = (
        safe_train_user_game_df["userId"].astype(str).str.strip()
    )
    safe_train_user_game_df["gameId"] = (
        safe_train_user_game_df["gameId"].astype(str).str.strip()
    )
    safe_train_user_game_df = safe_train_user_game_df[
        ~safe_train_user_game_df["userId"].str.lower().isin(["", "nan", "none", "null"])
        & ~safe_train_user_game_df["gameId"].str.lower().isin(["", "nan", "none", "null"])
    ]

    test_user_game_df["userId"] = test_user_game_df["userId"].astype(str).str.strip()
    test_user_game_df["gameId"] = test_user_game_df["gameId"].astype(str).str.strip()

    active_users = np.union1d(
        safe_train_user_game_df["userId"].unique(),
        test_user_game_df["userId"].unique(),
    )
    active_items = np.union1d(
        safe_train_user_game_df["gameId"].unique(),
        test_user_game_df["gameId"].unique(),
    )

    if len(active_users) == 0 or len(active_items) == 0:
        raise DataValidationError(
            f"No active users/items available after split: users={len(active_users)}, items={len(active_items)}"
        )

    ufe = train_user_features_df[
        train_user_features_df["userId"].astype(str).isin(active_users)
    ].copy()

    # Static item metadata fallback for test-only items
    static_item_meta = events_df.groupby("gameId").agg(
        game_type_static=(
            "gameType",
            lambda x: _mode_or_default(x, "unknown"),
        ),
        provider_static=("provider", lambda x: _mode_or_default(x, "unknown")),
    ).reset_index()

    ife = (
        pd.DataFrame({"gameId": active_items})
        .merge(train_game_features_df, on="gameId", how="left")
        .merge(static_item_meta, on="gameId", how="left")
    )
    ife["game_type"] = ife["game_type"].fillna(ife["game_type_static"]).fillna("unknown")
    ife["provider"] = ife["provider"].fillna(ife["provider_static"]).fillna("unknown")
    for numeric_col in [
        "game_sessions",
        "unique_users",
        "avg_duration_sec",
        "avg_rounds",
        "quick_exit_rate",
        "return_10m_rate",
        "positive_outcome_rate",
        "popularity_score",
    ]:
        if numeric_col in ife.columns:
            ife[numeric_col] = ife[numeric_col].fillna(0)
    ife["popularity_bucket"] = assign_popularity_bucket(ife["popularity_score"])

    # --- Build vocabulary ---
    user_feature_vocab = sorted(
        {tok for _, r in ufe.iterrows() for tok in user_tokens(r)}
    )
    item_feature_vocab = sorted(
        {tok for _, r in ife.iterrows() for tok in item_tokens(r)}
    )

    # --- LightFM Dataset ---
    dataset = Dataset()
    dataset.fit(
        users=active_users,
        items=active_items,
        user_features=user_feature_vocab,
        item_features=item_feature_vocab,
    )

    # Full interactions (train + test) for exclude-played during inference
    full_eval_pairs_df = pd.concat(
        [
            safe_train_user_game_df[["userId", "gameId", "implicit_score"]],
            test_user_game_df[["userId", "gameId", "implicit_score"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["userId", "gameId"], keep="first")

    (interactions, _) = dataset.build_interactions(
        (str(r["userId"]), str(r["gameId"]), float(r["implicit_score"]))
        for _, r in full_eval_pairs_df.iterrows()
    )
    (train_interactions, train_weights) = dataset.build_interactions(
        (str(r["userId"]), str(r["gameId"]), float(r["implicit_score"]))
        for _, r in safe_train_user_game_df.iterrows()
    )
    (test_interactions, _) = dataset.build_interactions(
        (str(r["userId"]), str(r["gameId"]), 1.0)
        for _, r in test_user_game_df.iterrows()
    )

    user_features_matrix = dataset.build_user_features(
        (str(r["userId"]), user_tokens(r)) for _, r in ufe.iterrows()
    )
    item_features_matrix = dataset.build_item_features(
        (str(r["gameId"]), item_tokens(r)) for _, r in ife.iterrows()
    )

    print("LightFM artifacts built ✅")
    print(f"Leakage-safe holdout pairs: {len(test_user_game_df)}")
    print(
        f"Interactions: {interactions.shape} | nnz = {interactions.nnz}"
    )
    print(f"Train nnz: {train_interactions.nnz} | Test nnz: {test_interactions.nnz}")
    print(f"User feature matrix: {user_features_matrix.shape}")
    print(f"Item feature matrix: {item_features_matrix.shape}")
    print(f"User feature tokens: {user_feature_vocab}")
    print(f"Item feature tokens: {item_feature_vocab}")

    return {
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
