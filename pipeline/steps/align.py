import re

import numpy as np
import pandas as pd
from prefect import task

from pipeline.exceptions import DataValidationError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)

SCHEMA_COLUMNS = [
    "eventType",
    "userId",
    "sessionId",
    "gameId",
    "gameType",
    "provider",
    "timestamp",
    "durationSeconds",
    "roundsPlayed",
    "stakeLevelCategory",
    "outcome",
    "exitType",
    "returnedWithin10mins",
    "deviceType",
    "timeOfDay",
    "dayOfWeek",
    "entryPoint",
]

VALID_GAME_TYPES = ["slot", "table", "live_dealer", "crash", "instant_win", "unknown"]
VALID_PROVIDERS = [
    "HUB88",
    "Evolution",
    "EZUGI",
    "Spribe",
    "OnlyPlay",
    "PeterAndSons",
    "unknown",
]
VALID_DEVICE_TYPES = ["mobile", "desktop", "tablet", "unknown"]
VALID_ENTRY_POINTS = [
    "livecasino",
    "casino",
    "sportsbook",
    "direct",
    "search",
    "lobby",
    "unknown",
]

OBJECT_ID_LIKE_RE = re.compile(r"^[0-9a-f]{24}$")


def is_objectid_like(raw_value) -> bool:
    value = str(raw_value).strip().lower()
    return bool(OBJECT_ID_LIKE_RE.fullmatch(value))


def map_game_type(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if any(
        k in value
        for k in [
            "live",
            "baccarat",
            "blackjack",
            "roulette",
            "sicbo",
            "sic bo",
            "dragon",
            "teen patti",
            "andar",
            "football studio",
            "fan tan",
            "bac bo",
        ]
    ):
        return "live_dealer"
    if any(k in value for k in ["slot", "starburst", "fruit", "reel"]):
        return "slot"
    if any(k in value for k in ["table", "poker"]):
        return "table"
    if any(k in value for k in ["crash", "aviator", "cricket crash", "quantum x", "cosmox"]):
        return "crash"
    if any(k in value for k in ["instant", "scratch", "keno"]):
        return "instant_win"
    return "unknown"


def map_provider(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if value in ("", "none", "nan", "null") or is_objectid_like(value):
        return "unknown"
    if any(k in value for k in ["evolution", "evosw", "evo"]):
        return "Evolution"
    if "ezugi" in value:
        return "EZUGI"
    if "spribe" in value:
        return "Spribe"
    if "onlyplay" in value:
        return "OnlyPlay"
    if "hub88" in value:
        return "HUB88"
    if "peterandsons" in value or "peter & sons" in value:
        return "PeterAndSons"
    # Preserve raw vendor name if not in known set and not an opaque id
    return str(raw_value).strip()


def map_device_type(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if any(k in value for k in ["android", "ios", "mobile", "phone"]):
        return "mobile"
    if any(k in value for k in ["desktop", "web", "windows", "mac", "linux", "pc"]):
        return "desktop"
    if "tablet" in value:
        return "tablet"
    return "unknown"


def map_entry_point(raw_value: str) -> str:
    value = str(raw_value).strip().lower().replace(" ", "_")
    if value in ("", "none", "nan", "null"):
        return "unknown"
    if "live" in value:
        return "livecasino"
    if "sport" in value:
        return "sportsbook"
    if value in ("home", "direct", "lobby"):
        return value if value != "home" else "direct"
    if "casino" in value:
        return "casino"
    # Preserve real source values like search, promo, campaign, etc.
    return value


def map_time_of_day(ts: pd.Series) -> pd.Series:
    hour = ts.dt.hour
    out = np.select(
        [hour.between(6, 11), hour.between(12, 17), hour.between(18, 21)],
        ["morning", "afternoon", "evening"],
        default="late_night",
    )
    return pd.Series(out, index=ts.index)


def map_day_of_week(ts: pd.Series) -> pd.Series:
    return ts.dt.day_name().str.lower()


def map_outcome(raw_result: pd.Series) -> pd.Series:
    value = raw_result.astype(str).str.upper()
    return np.where(
        value.eq("WIN"),
        "net_positive",
        np.where(value.isin(["LOSS", "LOSE"]), "net_negative", "break_even"),
    )


@task
def align_to_schema(transactions_df: pd.DataFrame) -> pd.DataFrame:
    df = transactions_df.copy()
    if df.empty:
        raise DataValidationError("Cannot align empty transactions dataframe to schema.")
    logger.info("Aligning %s transaction rows to event schema", len(df))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "_id" in df.columns:
        session_id = df["_id"].astype(str).map(lambda x: f"sess_{x}")
    else:
        session_id = pd.Series(range(len(df)), index=df.index).map(
            lambda x: f"sess_txn_{x}"
        )

    if "gameId" in df.columns:
        game_id = df["gameId"].astype(str)
    else:
        game_id = (
            "game_"
            + df["gameName"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )

    if "session_duration_sec" in df.columns:
        duration_seconds = pd.to_numeric(
            df["session_duration_sec"], errors="coerce"
        ).fillna(0)
    else:
        duration_seconds = pd.Series(0, index=df.index)

    rounds_played = pd.Series(1, index=df.index)

    stake = pd.to_numeric(df["betAmount"], errors="coerce").fillna(0)
    q1, q2 = stake.quantile([0.33, 0.66])
    stake_level = np.select(
        [stake <= q1, stake <= q2],
        ["low", "medium"],
        default="high",
    )

    entry_raw = df.get(
        "entryPoint_raw", pd.Series(index=df.index, dtype="object")
    ).fillna("unknown")
    device_raw = df.get(
        "deviceType_raw", pd.Series(index=df.index, dtype="object")
    ).fillna("unknown")

    aligned = pd.DataFrame(
        {
            "eventType": "game_session",
            "userId": df["userId"].astype(str),
            "sessionId": session_id,
            "gameId": game_id,
            "gameType": df.get("categoryName", df.get("gameType", "unknown")).map(
                map_game_type
            ),
            "provider": df.get("providerName", "unknown").map(map_provider),
            "timestamp": df["timestamp"],
            "durationSeconds": duration_seconds.astype(float),
            "roundsPlayed": rounds_played.astype(int),
            "stakeLevelCategory": stake_level,
            "outcome": map_outcome(
                df.get("result", pd.Series("break_even", index=df.index))
            ),
            "deviceType": device_raw.map(map_device_type),
            "entryPoint": entry_raw.map(map_entry_point),
        }
    )

    # Strong ID sanitization
    aligned["userId"] = aligned["userId"].astype(str).str.strip()
    aligned["gameId"] = aligned["gameId"].astype(str).str.strip()

    invalid_user = aligned["userId"].str.lower().isin({"", "nan", "none", "null"})
    invalid_game = aligned["gameId"].str.lower().isin({"", "nan", "none", "null"})
    invalid_time = aligned["timestamp"].isna()
    drop_mask = invalid_user | invalid_game | invalid_time
    dropped = int(drop_mask.sum())
    if dropped > 0:
        aligned = aligned.loc[~drop_mask].copy()
        print(f"Dropped invalid events: {dropped}")

    aligned = aligned.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    next_ts = aligned.groupby("userId")["timestamp"].shift(-1)
    gap_mins = (next_ts - aligned["timestamp"]).dt.total_seconds().div(60)
    aligned["returnedWithin10mins"] = gap_mins.le(10).fillna(False)

    aligned["exitType"] = np.select(
        [
            aligned["durationSeconds"] == 0,
            aligned["durationSeconds"] < 30,
            aligned["returnedWithin10mins"],
        ],
        ["unknown", "quick_exit", "returned_quickly"],
        default="natural_end",
    )

    aligned["timeOfDay"] = map_time_of_day(aligned["timestamp"])
    aligned["dayOfWeek"] = map_day_of_week(aligned["timestamp"])

    aligned = aligned[SCHEMA_COLUMNS]

    print(f"Events shape: {aligned.shape}")
    print(f"Schema columns present: {set(SCHEMA_COLUMNS).issubset(aligned.columns)}")
    if aligned.empty:
        raise DataValidationError("No valid events remained after schema alignment.")
    return aligned
