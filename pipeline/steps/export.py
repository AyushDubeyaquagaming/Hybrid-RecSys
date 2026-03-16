import json
import os

import joblib
import pandas as pd
from lightfm import LightFM
from prefect import task

from pipeline.config import PipelineSettings
from pipeline.exceptions import ArtifactExportError, DataValidationError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


@task
def export_artifacts(
    model: LightFM,
    dataset_artifacts: dict,
    games_df: pd.DataFrame,
    settings: PipelineSettings,
) -> str:
    artifact_dir = settings.ARTIFACT_DIR
    logger.info("Exporting artifacts to %s", artifact_dir)

    try:
        os.makedirs(artifact_dir, exist_ok=True)

        dataset = dataset_artifacts["dataset"]
        interactions = dataset_artifacts["interactions"]
        user_features_matrix = dataset_artifacts["user_features_matrix"]
        item_features_matrix = dataset_artifacts["item_features_matrix"]
        ife = dataset_artifacts["ife"]

        if interactions.nnz == 0:
            raise DataValidationError("Refusing to export empty interactions matrix.")

        # 1. Trained model
        joblib.dump(model, f"{artifact_dir}/model.joblib")

        # 2. LightFM Dataset object
        joblib.dump(dataset, f"{artifact_dir}/dataset.joblib")

        # 3. Feature matrices
        joblib.dump(user_features_matrix, f"{artifact_dir}/user_features_matrix.joblib")
        joblib.dump(item_features_matrix, f"{artifact_dir}/item_features_matrix.joblib")

        # 4. Full observed interactions (for exclude_played)
        # IMPORTANT: Use `interactions` (full matrix), NOT `train_interactions`.
        joblib.dump(interactions, f"{artifact_dir}/interactions.joblib")

        # 5. ID mappings
        user_id_map_raw, _, item_id_map_raw, _ = dataset.mapping()
        user_id_map_export = {str(k): int(v) for k, v in user_id_map_raw.items()}
        item_id_map_export = {str(k): int(v) for k, v in item_id_map_raw.items()}
        with open(f"{artifact_dir}/user_id_map.json", "w") as f:
            json.dump(user_id_map_export, f)
        with open(f"{artifact_dir}/item_id_map.json", "w") as f:
            json.dump(item_id_map_export, f)

        # 6. Game metadata for response enrichment
        games_name_lookup = {}
        for _, row in games_df.iterrows():
            gid = str(row["gameId"]).split(".")[0] if pd.notna(row["gameId"]) else None
            if gid:
                games_name_lookup[gid] = row.get("gameName", gid)

        game_meta = {}
        for _, row in ife.iterrows():
            gid = str(row["gameId"])
            game_meta[gid] = {
                "gameName": games_name_lookup.get(gid, gid),
                "gameType": str(row.get("game_type", "unknown")),
                "provider": str(row.get("provider", "unknown")),
            }
        with open(f"{artifact_dir}/game_metadata.json", "w") as f:
            json.dump(game_meta, f, indent=2)

        # 7. Popularity ranking for cold-start fallback
        popularity_df = ife[["gameId", "popularity_score"]].copy()
        popularity_df = popularity_df.sort_values("popularity_score", ascending=False)
        popularity_ranking = [str(gid) for gid in popularity_df["gameId"].tolist()]
        with open(f"{artifact_dir}/popularity_ranking.json", "w") as f:
            json.dump(popularity_ranking, f, indent=2)

    except Exception as exc:
        raise ArtifactExportError(f"Failed to export artifacts to {artifact_dir}: {exc}") from exc

    print(f"Artifacts exported to {artifact_dir}/")
    print(f"  model.joblib")
    print(f"  dataset.joblib")
    print(f"  user_features_matrix.joblib")
    print(f"  item_features_matrix.joblib")
    print(
        f"  interactions.joblib "
        f"(shape: {interactions.shape}, nnz: {interactions.nnz})"
    )
    print(f"  user_id_map.json ({len(user_id_map_export)} users)")
    print(f"  item_id_map.json ({len(item_id_map_export)} items)")
    print(f"  game_metadata.json ({len(game_meta)} games)")
    print(f"  popularity_ranking.json ({len(popularity_ranking)} games)")

    return artifact_dir


@task
def write_features_to_redis(
    dataset_artifacts: dict, settings: PipelineSettings
) -> int:
    """Write user and game features to Redis HashMaps.

    Best-effort: logs a warning and returns 0 on any failure.
    Returns total keys written.
    """
    if not settings.REDIS_ENABLED:
        print("Redis disabled, skipping feature store write")
        return 0

    try:
        import redis

        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD or None,
            decode_responses=True,
        )
        r.ping()  # fail fast if Redis is down

        ufe = dataset_artifacts["ufe"]
        ife = dataset_artifacts["ife"]
        ttl = settings.REDIS_TTL_SECONDS
        keys_written = 0

        # Write user features
        for _, row in ufe.iterrows():
            key = f"user:{row['userId']}:features"
            feature_map = {
                "total_sessions": str(row.get("total_sessions", 0)),
                "avg_duration_sec": str(round(row.get("avg_duration_sec", 0), 2)),
                "unique_games": str(row.get("unique_games", 0)),
                "unique_providers": str(row.get("unique_providers", 0)),
                "quick_exit_rate": str(round(row.get("quick_exit_rate", 0), 4)),
                "return_10m_rate": str(round(row.get("return_10m_rate", 0), 4)),
                "positive_outcome_rate": str(round(row.get("positive_outcome_rate", 0), 4)),
                "avg_engagement_intensity": str(round(row.get("avg_engagement_intensity", 0), 4)),
                "preferred_time_of_day": str(row.get("preferred_time_of_day", "unknown")),
                "preferred_day_of_week": str(row.get("preferred_day_of_week", "unknown")),
                "preferred_device": str(row.get("preferred_device", "unknown")),
                "preferred_entry_point": str(row.get("preferred_entry_point", "unknown")),
                "recency_days": str(round(row.get("recency_days", 0), 2)),
            }
            r.hset(key, mapping=feature_map)
            r.expire(key, ttl)
            keys_written += 1

        # Write game features
        for _, row in ife.iterrows():
            key = f"game:{row['gameId']}:features"
            feature_map = {
                "gameId": str(row["gameId"]),
                "gameType": str(row.get("game_type", "unknown")),
                "provider": str(row.get("provider", "unknown")),
                "game_sessions": str(row.get("game_sessions", 0)),
                "unique_users": str(row.get("unique_users", 0)),
                "popularity_score": str(round(row.get("popularity_score", 0), 4)),
                "popularity_bucket": str(row.get("popularity_bucket", "cold")),
                "return_10m_rate": str(round(row.get("return_10m_rate", 0), 4)),
                "positive_outcome_rate": str(round(row.get("positive_outcome_rate", 0), 4)),
            }
            r.hset(key, mapping=feature_map)
            r.expire(key, ttl)
            keys_written += 1

        print(f"Redis feature store: wrote {keys_written} keys (TTL={ttl}s)")
        logger.info("Redis feature store: wrote %s keys", keys_written)
        return keys_written

    except Exception as exc:
        print(f"WARNING: Redis write failed (best-effort): {exc}")
        logger.warning("Redis write failed (best-effort): %s", exc)
        return 0
