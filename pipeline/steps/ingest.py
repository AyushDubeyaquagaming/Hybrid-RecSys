import pandas as pd
import numpy as np
from prefect import task

from pipeline.config import PipelineSettings
from pipeline.db import get_db as _get_db
from pipeline.exceptions import DataValidationError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


def extract_bet_part(betParts):
    if betParts and len(betParts) > 0:
        part = betParts[0]
        return {
            "sportName": part.get("sportName", None),
            "categoryName": part.get("categoryName", None),
            "tournamentName": part.get("tournamentName", None),
            "tournamentId": part.get("tournamentId", None),
        }
    return {
        "sportName": None,
        "categoryName": None,
        "tournamentName": None,
        "tournamentId": None,
    }


@task
def load_transactions(settings: PipelineSettings) -> pd.DataFrame:
    db = _get_db(settings)
    raw_transactions = list(
        db["bet_transactions"].find(
            {
                "gameType": settings.GAME_TYPE_FILTER,
                "status": settings.BET_STATUS_FILTER,
                "betParts": {"$exists": True, "$ne": []},
            }
        )
    )
    transactions_df = pd.DataFrame(raw_transactions)
    logger.info("Loaded raw transactions: %s", len(transactions_df))
    print(f"Raw transactions loaded: {len(transactions_df)}")

    if transactions_df.empty:
        raise DataValidationError(
            "No transactions returned from bet_transactions for the configured filters."
        )
    if "betParts" not in transactions_df.columns:
        raise DataValidationError("Expected column 'betParts' missing from transactions data.")

    bet_parts_expanded = transactions_df["betParts"].apply(extract_bet_part)
    bet_parts_df = pd.DataFrame(bet_parts_expanded.tolist())
    transactions_df = pd.concat(
        [transactions_df.drop(columns=["betParts"]), bet_parts_df], axis=1
    )

    transactions_df = transactions_df.rename(
        columns={
            "loginId": "userId",
            "stake": "betAmount",
            "createdDate": "timestamp",
            "sportName": "gameName",
            "tournamentName": "providerName_raw",
        }
    )

    transactions_df["timestamp"] = pd.to_datetime(
        transactions_df["timestamp"], errors="coerce"
    )
    transactions_df["hourOfDay"] = transactions_df["timestamp"].dt.hour
    transactions_df["dayOfWeek"] = transactions_df["timestamp"].dt.day_name()

    transactions_df = transactions_df[
        [
            "userId",
            "gameName",
            "categoryName",
            "providerName_raw",
            "tournamentId",
            "betAmount",
            "result",
            "timestamp",
            "hourOfDay",
            "dayOfWeek",
        ]
    ].copy()

    transactions_df = transactions_df.dropna(subset=["userId", "gameName", "betAmount"])
    if transactions_df.empty:
        raise DataValidationError(
            "All transactions were dropped after cleaning; no usable training rows remain."
        )
    print(f"Clean transactions: {len(transactions_df)}")
    return transactions_df


@task
def load_game_details(settings: PipelineSettings) -> pd.DataFrame:
    db = _get_db(settings)
    games_raw = list(
        db["gamedetails"].find(
            {"gameStatus": "ON"},
            {
                "gameId": 1,
                "gameName": 1,
                "minBet": 1,
                "maxBet": 1,
                "category": 1,
                "gamevendor": 1,
            },
        )
    )
    games_df = pd.DataFrame(games_raw)
    logger.info("Loaded game details: %s", len(games_df))
    if games_df.empty:
        raise DataValidationError("No active games returned from gamedetails.")
    if "category" not in games_df.columns:
        raise DataValidationError("Expected column 'category' missing from gamedetails data.")
    games_df["gameType"] = games_df["category"].apply(
        lambda x: x[0]["name"] if x and len(x) > 0 else "Unknown"
    )
    games_df = games_df[
        ["gameId", "gameName", "gameType", "gamevendor", "minBet", "maxBet"]
    ]
    print(f"Games loaded: {len(games_df)}")
    return games_df


@task
def load_users(settings: PipelineSettings) -> pd.DataFrame:
    db = _get_db(settings)
    users_raw = list(
        db["players"].find(
            {"activeStatus": True},
            {"playerId": 1, "username": 1, "contactNo": 1, "activeStatus": 1},
        )
    )
    users_df = pd.DataFrame(users_raw)
    logger.info("Loaded users: %s", len(users_df))
    if users_df.empty:
        logger.warning("No active users returned from players; downstream device enrichment may have limited coverage.")
        return pd.DataFrame(columns=["userId", "playerId", "activeStatus"])
    users_df = users_df.rename(columns={"contactNo": "userId"})
    users_df = users_df[["userId", "playerId", "activeStatus"]]
    print(f"Users loaded: {len(users_df)}")
    return users_df


@task
def clean_and_merge(
    transactions_df: pd.DataFrame,
    games_df: pd.DataFrame,
    settings: PipelineSettings,
) -> pd.DataFrame:
    transactions_df = transactions_df.copy()
    logger.info(
        "Cleaning and merging transactions=%s with games=%s",
        len(transactions_df),
        len(games_df),
    )

    # Normalize game names
    transactions_df["gameName"] = transactions_df["gameName"].replace(
        settings.GAME_NAME_MAP
    )

    # Merge gameId + gamevendor from games_df (deduplicated to prevent row inflation)
    pre_merge_len = len(transactions_df)
    transactions_df = transactions_df.merge(
        games_df[["gameName", "gameId", "gamevendor", "minBet", "maxBet"]].drop_duplicates(
            "gameName"
        ),
        on="gameName",
        how="left",
    )
    if len(transactions_df) != pre_merge_len:
        raise DataValidationError(
            f"Merge inflated rows: {pre_merge_len} -> {len(transactions_df)}"
        )

    # Provider source priority: betParts.tournamentName -> gamedetails.gamevendor fallback
    transactions_df["providerName"] = transactions_df["providerName_raw"].combine_first(
        transactions_df["gamevendor"]
    )
    transactions_df = transactions_df.drop(columns=["gamevendor"], errors="ignore")

    # Binary win flag
    transactions_df["win"] = (transactions_df["result"] == "WIN").astype(int)

    print(f"After cleaning: {len(transactions_df)}")
    print(
        f"Games with gameId matched: {transactions_df['gameId'].notna().sum()}"
    )
    print(
        f"Provider coverage: {transactions_df['providerName'].notna().sum()}/{len(transactions_df)}"
    )
    return transactions_df
