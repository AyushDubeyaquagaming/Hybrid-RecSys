"""
Unit tests for ModelService prediction logic.

Uses real (small) LightFM artifacts built in-memory — no files needed.
"""
import json
import os
import tempfile

import joblib
import numpy as np
import pytest
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix


def build_artifacts(tmpdir: str):
    """Build a tiny but real LightFM model + artifacts for testing."""
    users = ["u1", "u2", "u3"]
    items = ["g1", "g2", "g3", "g4", "g5"]

    ds = Dataset()
    ds.fit(users=users, items=items)

    (interactions, _) = ds.build_interactions(
        [
            ("u1", "g1"),
            ("u1", "g2"),
            ("u2", "g3"),
            ("u3", "g4"),
            ("u3", "g5"),
        ]
    )

    user_features_matrix, item_features_matrix = None, None
    # Build identity feature matrices (just IDs as features)
    ds.fit_partial(
        user_features=[f"uf_{u}" for u in users],
        item_features=[f"if_{i}" for i in items],
    )
    user_features_matrix = ds.build_user_features(
        [(u, [f"uf_{u}"]) for u in users]
    )
    item_features_matrix = ds.build_item_features(
        [(i, [f"if_{i}"]) for i in items]
    )

    mdl = LightFM(no_components=4, loss="bpr")
    mdl.fit(
        interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        epochs=3,
        num_threads=1,
    )

    joblib.dump(mdl, os.path.join(tmpdir, "model.joblib"))
    joblib.dump(ds, os.path.join(tmpdir, "dataset.joblib"))
    joblib.dump(user_features_matrix, os.path.join(tmpdir, "user_features_matrix.joblib"))
    joblib.dump(item_features_matrix, os.path.join(tmpdir, "item_features_matrix.joblib"))
    joblib.dump(interactions, os.path.join(tmpdir, "interactions.joblib"))

    user_id_map_raw, _, item_id_map_raw, _ = ds.mapping()
    user_id_map = {str(k): int(v) for k, v in user_id_map_raw.items()}
    item_id_map = {str(k): int(v) for k, v in item_id_map_raw.items()}

    json.dump(user_id_map, open(os.path.join(tmpdir, "user_id_map.json"), "w"))
    json.dump(item_id_map, open(os.path.join(tmpdir, "item_id_map.json"), "w"))

    game_meta = {
        item: {"gameName": f"Game {item}", "gameType": "slots", "provider": "TestCo"}
        for item in items
    }
    json.dump(game_meta, open(os.path.join(tmpdir, "game_metadata.json"), "w"))

    # popularity ranking: just reverse alphabetical for determinism
    popularity_ranking = list(reversed(items))
    json.dump(popularity_ranking, open(os.path.join(tmpdir, "popularity_ranking.json"), "w"))

    return {"users": users, "items": items}


@pytest.fixture(scope="module")
def artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = build_artifacts(tmpdir)
        yield tmpdir, meta


@pytest.fixture(scope="module")
def service(artifacts):
    from app.services.model_service import ModelService

    tmpdir, _ = artifacts
    svc = ModelService()
    svc.load_artifacts(tmpdir, num_threads=1)
    return svc


def test_load_artifacts(service):
    assert service.is_loaded() is True


def test_predict_returns_correct_count(service):
    result = service.recommend("u1", top_k=3, exclude_played=True)
    assert len(result["recommendations"]) <= 3
    assert len(result["recommendations"]) > 0


def test_cold_start_returns_popular(service, artifacts):
    _, meta = artifacts
    result = service.recommend("unknown_user_xyz", top_k=3, exclude_played=True)
    assert result["is_cold_start"] is True
    assert result["source"] == "popularity_fallback"
    # Should return games in popularity order (reversed items list)
    expected_order = list(reversed(meta["items"]))
    returned_ids = [r["game_id"] for r in result["recommendations"]]
    assert returned_ids == expected_order[: len(returned_ids)]


def test_scores_descending(service):
    result = service.recommend("u1", top_k=5, exclude_played=False)
    scores = [r["score"] for r in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_exclude_played_removes_known(service):
    result_excl = service.recommend("u1", top_k=5, exclude_played=True)
    played_ids = {"g1", "g2"}  # u1 played g1 and g2
    returned_ids = {r["game_id"] for r in result_excl["recommendations"]}
    assert returned_ids.isdisjoint(played_ids)


def test_include_played_allows_known(service):
    result_incl = service.recommend("u1", top_k=5, exclude_played=False)
    returned_ids = {r["game_id"] for r in result_incl["recommendations"]}
    # With all 5 items and exclude_played=False, all items should appear
    assert len(returned_ids) == 5
