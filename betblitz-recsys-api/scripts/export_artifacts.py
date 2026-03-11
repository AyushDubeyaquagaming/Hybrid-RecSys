"""
Run this script inside the notebook environment after all cells have executed.
All variables referenced here exist in the notebook's global scope after a full run.

Usage (from within the notebook kernel or via %run):
    %run scripts/export_artifacts.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- 1. Trained LightFM model ---
joblib.dump(model, f"{ARTIFACT_DIR}/model.joblib")

# --- 2. LightFM Dataset object ---
joblib.dump(dataset, f"{ARTIFACT_DIR}/dataset.joblib")

# --- 3. Feature matrices (sparse, precomputed) ---
joblib.dump(user_features_matrix, f"{ARTIFACT_DIR}/user_features_matrix.joblib")
joblib.dump(item_features_matrix, f"{ARTIFACT_DIR}/item_features_matrix.joblib")

# --- 4. Full observed interactions (for exclude_played) ---
# IMPORTANT: Use `interactions` (full matrix), NOT `train_interactions`.
# Product meaning of "exclude played" = all games the user has ever touched.
# This matches the notebook's recommend_for_user() which masks from observed_csr.
joblib.dump(interactions, f"{ARTIFACT_DIR}/interactions.joblib")

# --- 5. ID mappings ---
user_id_map_raw, _, item_id_map_raw, _ = dataset.mapping()
# Do NOT cast with int() — gameId can be non-numeric fallback IDs like "game_zombies"
user_id_map_export = {str(k): int(v) for k, v in user_id_map_raw.items()}
item_id_map_export = {str(k): int(v) for k, v in item_id_map_raw.items()}
json.dump(user_id_map_export, open(f"{ARTIFACT_DIR}/user_id_map.json", "w"))
json.dump(item_id_map_export, open(f"{ARTIFACT_DIR}/item_id_map.json", "w"))

# --- 6. Game metadata for response enrichment ---
# Pull gameName from games_df (the gamedetails collection), not from events_df
game_meta = {}
# First, index games_df by gameId for lookup
games_name_lookup = {}
if "games_df" in globals():
    for _, row in games_df.iterrows():
        gid = str(row["gameId"]).split(".")[0] if pd.notna(row["gameId"]) else None
        if gid:
            games_name_lookup[gid] = row.get("gameName", gid)

# Build metadata from game features (which has game_type, provider)
# ife is the item features table used during training
for _, row in ife.iterrows():
    gid = str(row["gameId"])
    game_meta[gid] = {
        "gameName": games_name_lookup.get(gid, gid),
        "gameType": str(row.get("game_type", "unknown")),
        "provider": str(row.get("provider", "unknown")),
    }

json.dump(game_meta, open(f"{ARTIFACT_DIR}/game_metadata.json", "w"), indent=2)

# --- 7. Popularity ranking for cold-start fallback ---
# Use popularity_score from game features (log1p of session count)
# Sort descending — most popular first
popularity_df = ife[["gameId", "popularity_score"]].copy()
popularity_df = popularity_df.sort_values("popularity_score", ascending=False)
popularity_ranking = [str(gid) for gid in popularity_df["gameId"].tolist()]
json.dump(
    popularity_ranking, open(f"{ARTIFACT_DIR}/popularity_ranking.json", "w"), indent=2
)

print(f"Artifacts exported to {ARTIFACT_DIR}/")
print(f"  model.joblib")
print(f"  dataset.joblib")
print(f"  user_features_matrix.joblib")
print(f"  item_features_matrix.joblib")
print(
    f"  interactions.joblib (shape: {interactions.shape}, nnz: {interactions.nnz})"
)
print(f"  user_id_map.json ({len(user_id_map_export)} users)")
print(f"  item_id_map.json ({len(item_id_map_export)} items)")
print(f"  game_metadata.json ({len(game_meta)} games)")
print(
    f"  popularity_ranking.json ({len(popularity_ranking)} games, most popular first)"
)
