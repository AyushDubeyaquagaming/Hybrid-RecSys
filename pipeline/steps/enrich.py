import pandas as pd
import numpy as np
from pymongo import MongoClient
from prefect import task

from pipeline.config import PipelineSettings
from pipeline.exceptions import DataValidationError, ExternalServiceError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


def _get_db(settings: PipelineSettings):
    try:
        client = MongoClient(
            settings.MONGO_URI,
            directConnection=settings.MONGO_DIRECT_CONNECTION,
            serverSelectionTimeoutMS=settings.MONGO_TIMEOUT_MS,
        )
        client.admin.command("ping")
        return client[settings.MONGO_DB]
    except Exception as exc:
        raise ExternalServiceError(
            f"Failed to connect to MongoDB at {settings.MONGO_URI}: {exc}"
        ) from exc


@task
def enrich_sessions(
    transactions_df: pd.DataFrame, settings: PipelineSettings
) -> pd.DataFrame:
    """Mirror notebook cell 14 exactly.

    Joins usersessionlog with transactions_df to populate
    session_duration_sec and entryPoint_raw.
    """
    SESSION_JOIN_TOLERANCE = pd.Timedelta(f"{settings.SESSION_JOIN_TOLERANCE_MIN}min")

    db = _get_db(settings)
    sessions_raw = list(
        db["usersessionlog"].find(
            {"status": "Closed", "source": settings.SESSION_SOURCE_FILTER},
            {
                "_id": 0,
                "user_id": 1,
                "game_id": 1,
                "source": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
    )
    sessions_df = pd.DataFrame(sessions_raw)
    logger.info("Loaded session logs for enrichment: %s", len(sessions_df))

    transactions_df = transactions_df.copy()
    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"], errors="coerce"
    )
    transactions_df = transactions_df.drop(
        columns=["session_duration_sec", "entryPoint_raw"], errors="ignore"
    )

    if len(sessions_df) == 0:
        transactions_df["session_duration_sec"] = np.nan
        transactions_df["entryPoint_raw"] = pd.Series(
            np.nan, index=transactions_df.index, dtype="object"
        )
        print(
            "No livecasino closed sessions found for enrichment. "
            "session_duration_sec/entryPoint_raw left as NaN"
        )
        return transactions_df

    sessions_df["session_duration_sec"] = (
        pd.to_datetime(sessions_df["updated_at"], errors="coerce")
        - pd.to_datetime(sessions_df["created_at"], errors="coerce")
    ).dt.total_seconds()

    sessions_df = sessions_df.rename(
        columns={
            "user_id": "userId",
            "game_id": "gameId",
            "source": "entryPoint_raw",
        }
    )

    sessions_df["userId"] = sessions_df["userId"].astype(str).str.strip()
    sessions_df["gameId"] = sessions_df["gameId"].astype(str).str.strip()
    sessions_df["updated_at"] = pd.to_datetime(sessions_df["updated_at"], errors="coerce")
    sessions_df = sessions_df.dropna(subset=["userId", "gameId", "updated_at"]).sort_values(
        ["updated_at", "userId", "gameId"]
    )

    transactions_df["userId"] = transactions_df["userId"].astype(str).str.strip()
    transactions_df["gameId_str"] = (
        transactions_df["gameId"].astype(str).str.split(".").str[0].str.strip()
    )

    tx_valid = transactions_df.dropna(subset=["timestamp"]).copy()
    tx_valid["_row_id"] = tx_valid.index
    tx_sorted = tx_valid.sort_values(["timestamp", "userId", "gameId_str"])

    sess_sorted = sessions_df.rename(
        columns={
            "session_duration_sec": "_session_duration_sec",
            "entryPoint_raw": "_entryPoint_raw",
        }
    )

    tx_enriched = pd.merge_asof(
        tx_sorted,
        sess_sorted[
            ["userId", "gameId", "updated_at", "_session_duration_sec", "_entryPoint_raw"]
        ],
        left_on="timestamp",
        right_on="updated_at",
        left_by=["userId", "gameId_str"],
        right_by=["userId", "gameId"],
        direction="backward",
        tolerance=SESSION_JOIN_TOLERANCE,
    )

    transactions_df["session_duration_sec"] = np.nan
    transactions_df["entryPoint_raw"] = pd.Series(
        np.nan, index=transactions_df.index, dtype="object"
    )
    transactions_df.loc[
        tx_enriched["_row_id"], "session_duration_sec"
    ] = tx_enriched["_session_duration_sec"].values
    transactions_df.loc[
        tx_enriched["_row_id"], "entryPoint_raw"
    ] = tx_enriched["_entryPoint_raw"].values
    transactions_df = transactions_df.drop(columns=["gameId_str"], errors="ignore")

    matched = transactions_df["session_duration_sec"].notna().sum()
    entry_matched = transactions_df["entryPoint_raw"].notna().sum()
    quick_exit_candidates = (
        transactions_df["session_duration_sec"].lt(30).sum() if matched else 0
    )
    print(f"Transactions with session duration: {matched}/{len(transactions_df)}")
    print(f"Session duration coverage: {matched/len(transactions_df)*100:.1f}%")
    print(f"Transactions with entryPoint_raw: {entry_matched}/{len(transactions_df)}")
    print(f"entryPoint coverage: {entry_matched/len(transactions_df)*100:.1f}%")
    print(f"Potential quick-exit sessions (<30s) retained: {quick_exit_candidates}")
    return transactions_df


@task
def enrich_device(
    transactions_df: pd.DataFrame,
    users_df: pd.DataFrame,
    settings: PipelineSettings,
) -> pd.DataFrame:
    """Mirror notebook cell 15 exactly.

    Joins useractivitylogs with transactions_df via a cascade of candidate
    key joins to populate deviceType_raw.
    """
    DEVICE_JOIN_TOLERANCE = pd.Timedelta(f"{settings.DEVICE_JOIN_TOLERANCE_HOURS}h")

    db = _get_db(settings)
    activity_raw = list(
        db["useractivitylogs"].find(
            {},
            {
                "_id": 0,
                "device_type": 1,
                "user_id": 1,
                "userId": 1,
                "loginId": 1,
                "playerId": 1,
                "created_at": 1,
                "updated_at": 1,
                "timestamp": 1,
                "event_time": 1,
            },
        )
    )
    activity_df = pd.DataFrame(activity_raw)
    logger.info("Loaded activity logs for enrichment: %s", len(activity_df))

    transactions_df = transactions_df.copy()
    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"], errors="coerce"
    )

    if len(activity_df) == 0:
        transactions_df["deviceType_raw"] = pd.Series(
            np.nan, index=transactions_df.index, dtype="object"
        )
        print("No records in useractivitylogs. deviceType_raw left as NaN")
        return transactions_df

    ts_col = None
    for c in ["updated_at", "created_at", "timestamp", "event_time"]:
        if c in activity_df.columns:
            ts_col = c
            break

    if ts_col is None:
        transactions_df["deviceType_raw"] = pd.Series(
            np.nan, index=transactions_df.index, dtype="object"
        )
        print("No timestamp column found in useractivitylogs. deviceType_raw left as NaN")
        return transactions_df

    activity_df["_activity_ts"] = pd.to_datetime(activity_df[ts_col], errors="coerce")
    activity_df["deviceType_raw"] = activity_df.get(
        "device_type", pd.Series(index=activity_df.index, dtype="object")
    )

    tx_device = transactions_df.copy()
    tx_device["userId"] = tx_device["userId"].astype(str).str.strip()
    tx_device["deviceType_raw"] = pd.Series(
        np.nan, index=tx_device.index, dtype="object"
    )

    if {"userId", "playerId"}.issubset(users_df.columns):
        player_bridge = users_df[["userId", "playerId"]].copy()
        player_bridge["userId"] = player_bridge["userId"].astype(str).str.strip()
        player_bridge["playerId_bridge"] = player_bridge["playerId"].astype(str).str.strip()
        tx_device = tx_device.merge(
            player_bridge[["userId", "playerId_bridge"]].drop_duplicates("userId"),
            on="userId",
            how="left",
        )
    else:
        tx_device["playerId_bridge"] = np.nan

    tx_valid_base = tx_device.dropna(subset=["timestamp"]).copy()
    tx_valid_base["_row_id"] = tx_valid_base.index

    candidate_joins = [
        ("userId", "playerId", "direct userId -> activity.playerId"),
        ("playerId_bridge", "playerId", "players bridge -> activity.playerId"),
        ("userId", "loginId", "direct userId -> activity.loginId"),
        ("userId", "user_id", "direct userId -> activity.user_id"),
        ("userId", "userId", "direct userId -> activity.userId"),
    ]

    join_reports = []

    for tx_key, act_key, label in candidate_joins:
        if tx_key not in tx_valid_base.columns or act_key not in activity_df.columns:
            continue

        tx_missing = tx_valid_base[
            tx_device.loc[tx_valid_base.index, "deviceType_raw"].isna()
        ].copy()
        if len(tx_missing) == 0:
            break

        tx_missing[tx_key] = tx_missing[tx_key].astype(str).str.strip()
        tx_missing = tx_missing[
            ~tx_missing[tx_key].str.lower().isin(["", "nan", "none", "null"])
        ].copy()
        if len(tx_missing) == 0:
            continue

        act_join = activity_df[[act_key, "_activity_ts", "deviceType_raw"]].copy()
        act_join[act_key] = act_join[act_key].astype(str).str.strip()
        act_join = act_join.dropna(subset=["_activity_ts"])
        act_join = act_join[
            ~act_join[act_key].str.lower().isin(["", "nan", "none", "null"])
        ].copy()
        if len(act_join) == 0:
            continue

        key_overlap = len(
            set(tx_missing[tx_key].unique()) & set(act_join[act_key].unique())
        )
        if key_overlap == 0:
            join_reports.append(f"{label}: 0 overlapping keys")
            continue

        tx_sorted = tx_missing.sort_values(["timestamp", tx_key])
        act_sorted = act_join.sort_values(["_activity_ts", act_key])

        tx_enriched = pd.merge_asof(
            tx_sorted,
            act_sorted,
            left_on="timestamp",
            right_on="_activity_ts",
            left_by=tx_key,
            right_by=act_key,
            direction="backward",
            tolerance=DEVICE_JOIN_TOLERANCE,
        )

        fill_series = tx_enriched.set_index("_row_id")["deviceType_raw_y"]
        matched_now = int(fill_series.notna().sum())
        join_reports.append(f"{label}: overlap={key_overlap}, matched={matched_now}")

        if matched_now > 0:
            tx_device["deviceType_raw"] = tx_device["deviceType_raw"].combine_first(
                fill_series
            )

    transactions_df["deviceType_raw"] = tx_device["deviceType_raw"]

    coverage = (
        transactions_df["deviceType_raw"].notna().mean() * 100
        if "deviceType_raw" in transactions_df.columns
        else 0
    )
    print("Device join diagnostics:")
    for report in join_reports:
        print("-", report)
    print(f"deviceType_raw coverage: {coverage:.1f}%")
    return transactions_df
