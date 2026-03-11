import json
import os

import joblib
import numpy as np
from scipy.sparse import csr_matrix


MODEL_VERSION = "v0.1-real-data"


class ModelService:
    def __init__(self):
        self._loaded = False
        self.model = None
        self.user_features_matrix = None
        self.item_features_matrix = None
        self.interactions_csr = None
        self.user_id_map: dict = {}       # userId (str) -> internal_idx (int)
        self.item_id_map: dict = {}       # gameId (str) -> internal_idx (int)
        self.idx_to_item: dict = {}       # internal_idx (int) -> gameId (str)
        self.game_metadata: dict = {}
        self.popularity_ranking: list = []
        self.n_users: int = 0
        self.n_items: int = 0
        self.num_threads: int = 1

    def load_artifacts(self, artifact_dir: str, num_threads: int = 1) -> None:
        self.num_threads = num_threads

        self.model = joblib.load(os.path.join(artifact_dir, "model.joblib"))

        self.user_features_matrix = joblib.load(
            os.path.join(artifact_dir, "user_features_matrix.joblib")
        )
        self.item_features_matrix = joblib.load(
            os.path.join(artifact_dir, "item_features_matrix.joblib")
        )

        interactions = joblib.load(os.path.join(artifact_dir, "interactions.joblib"))
        self.interactions_csr = csr_matrix(interactions)

        with open(os.path.join(artifact_dir, "user_id_map.json")) as f:
            self.user_id_map = json.load(f)

        with open(os.path.join(artifact_dir, "item_id_map.json")) as f:
            self.item_id_map = json.load(f)

        self.idx_to_item = {int(v): k for k, v in self.item_id_map.items()}

        with open(os.path.join(artifact_dir, "game_metadata.json")) as f:
            self.game_metadata = json.load(f)

        with open(os.path.join(artifact_dir, "popularity_ranking.json")) as f:
            self.popularity_ranking = json.load(f)

        self.n_users = self.interactions_csr.shape[0]
        self.n_items = self.interactions_csr.shape[1]

        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def recommend(self, user_id: str, top_k: int, exclude_played: bool) -> dict:
        if not self._loaded:
            raise RuntimeError("Model artifacts are not loaded")

        if user_id in self.user_id_map:
            user_idx = self.user_id_map[user_id]
            recs = self._predict_known_user(user_idx, top_k, exclude_played)
            return {
                "recommendations": recs,
                "is_cold_start": False,
                "source": "lightfm",
            }
        else:
            recs = self._popularity_fallback(top_k)
            return {
                "recommendations": recs,
                "is_cold_start": True,
                "source": "popularity_fallback",
            }

    def _predict_known_user(
        self, user_idx: int, top_k: int, exclude_played: bool
    ) -> list:
        n_items = self.n_items

        scores = self.model.predict(
            user_ids=np.repeat(user_idx, n_items),
            item_ids=np.arange(n_items),
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            num_threads=self.num_threads,
        )

        if exclude_played:
            known = self.interactions_csr[user_idx].indices
            scores[known] = -np.inf

        finite_mask = np.isfinite(scores)
        candidate_indices = np.where(finite_mask)[0]

        if len(candidate_indices) == 0:
            return []

        actual_k = min(top_k, len(candidate_indices))
        topk_of_candidates = np.argpartition(scores[candidate_indices], -actual_k)[
            -actual_k:
        ]
        topk_indices = candidate_indices[topk_of_candidates]
        topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]

        return [
            {
                "game_id": self.idx_to_item[i],
                "game_name": self.game_metadata.get(self.idx_to_item[i], {}).get(
                    "gameName", self.idx_to_item[i]
                ),
                "game_type": self.game_metadata.get(self.idx_to_item[i], {}).get(
                    "gameType", "unknown"
                ),
                "provider": self.game_metadata.get(self.idx_to_item[i], {}).get(
                    "provider", "unknown"
                ),
                "score": round(float(scores[i]), 4),
                "rank": rank + 1,
            }
            for rank, i in enumerate(topk_indices)
        ]

    def _popularity_fallback(self, top_k: int) -> list:
        results = []
        for rank, game_id in enumerate(self.popularity_ranking[:top_k]):
            meta = self.game_metadata.get(game_id, {})
            results.append(
                {
                    "game_id": game_id,
                    "game_name": meta.get("gameName", game_id),
                    "game_type": meta.get("gameType", "unknown"),
                    "provider": meta.get("provider", "unknown"),
                    "score": 0.0,
                    "rank": rank + 1,
                }
            )
        return results
