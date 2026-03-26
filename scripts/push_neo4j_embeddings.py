import json
from pathlib import Path

import joblib

from pipeline.config import PipelineSettings
from pipeline.steps.neo4j_export import export_embeddings_to_neo4j


def _ordered_ids_from_map(id_map: dict) -> list:
    ordered_ids = [None] * len(id_map)
    for raw_id, internal_id in id_map.items():
        ordered_ids[int(internal_id)] = raw_id

    missing = [index for index, value in enumerate(ordered_ids) if value is None]
    if missing:
        raise ValueError(f"ID map is not contiguous; missing indices: {missing}")

    return ordered_ids


def _load_dataset_artifacts(artifact_dir: Path) -> dict:
    user_map = json.loads((artifact_dir / "user_id_map.json").read_text())
    item_map = json.loads((artifact_dir / "item_id_map.json").read_text())

    return {
        "user_features_matrix": joblib.load(artifact_dir / "user_features_matrix.joblib"),
        "item_features_matrix": joblib.load(artifact_dir / "item_features_matrix.joblib"),
        "active_users": _ordered_ids_from_map(user_map),
        "active_items": _ordered_ids_from_map(item_map),
    }


def main() -> int:
    settings = PipelineSettings()
    artifact_dir = Path(settings.ARTIFACT_DIR)

    required_files = [
        artifact_dir / "model.joblib",
        artifact_dir / "user_features_matrix.joblib",
        artifact_dir / "item_features_matrix.joblib",
        artifact_dir / "user_id_map.json",
        artifact_dir / "item_id_map.json",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing artifact files required for Neo4j push: " + ", ".join(missing_files)
        )

    model = joblib.load(artifact_dir / "model.joblib")
    dataset_artifacts = _load_dataset_artifacts(artifact_dir)

    updated = export_embeddings_to_neo4j.fn(
        model,
        dataset_artifacts,
        settings,
        player_key=settings.NEO4J_PLAYER_KEY,
        game_key=settings.NEO4J_GAME_KEY,
    )
    print(f"Neo4j push completed with {updated} total node updates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())