# Live Casino Recommendation System
**Quantum Horizon — Phase 1 (Experimentation)**

---

## Overview

A hybrid collaborative + content-based recommendation system built with [LightFM](https://github.com/lyst/lightfm), designed to surface personalized live casino game recommendations for each user. This notebook covers the full experimentation pipeline: data simulation, EDA, feature engineering, model training, evaluation, and diagnostic analysis.

> **Status:** Proof-of-concept on simulated data. Data layer will be swapped to real MongoDB → pandas once the pipeline is ready.
> *Note: The values and plots shouldn't be considered as the true nature of the data since it's just a dummy data. Purely intended to maintain the overall flow of the notebook for experiment phase*
---

## Notebook Structure (Experimentation)

> Cells 6 & 7 were run as separate experimental blocks. In the dev pipeline below, they are consolidated into a single `analysis` stage.

| Cell | Description |
|------|-------------|
| 0 | Installs & Imports |
| 1 | Data Simulation (Games, Users, Transactions) |
| 2 | Basic EDA |
| 2B | Advanced EDA (Behavior Patterns & Data Geometry) |
| 3 | Feature Engineering (Interaction Matrix + User/Item Features) |
| 3B | Feature Ablation (Model Selection, Pre-Final Training) |
| 4 | LightFM Model Training |
| 5 | Final Evaluation (Precision@K, AUC, NDCG) |
| 5B | Ranking Confusion Matrix + PR Diagnostics |
| 6 | Post-Model Correlation Diagnostics (Feature Impact + Popularity Bias) |
| 7 | Model Output Visualizations |
| 8 | Auto Interpretation |

---

## Dev Pipeline Structure

How the above maps into a production-ready module layout once experimentation is complete.

| Stage | Module / File | What it does | Notebook source |
|-------|--------------|--------------|-----------------|
| **1. Data Ingestion** | `data/loader.py` | Loads users, games, transactions from MongoDB → pandas | Cell 1 *(replace simulation)* |
| **2. EDA** | `analysis/eda.py` | Summary stats, transaction distributions, game-type breakdowns | Cell 2 |
| **3. Analysis** | `analysis/advanced.py` | User behaviour, matrix sparsity, ablation study, feature–score correlation — all in one pass | Cells 6 + 7 *(merged)* |
| **4. Feature Engineering** | `features/build_features.py` | Implicit score, interaction matrix, user & item feature matrices | Cell 3 |
| **5. Training** | `model/train.py` | LightFM WARP training loop with epoch-level evaluation | Cell 4 |
| **6. Evaluation** | `model/evaluate.py` | Precision@K, AUC, decile ranking sanity check | Cell 5 |
| **7. Interpretation** | `reporting/interpret.py` | Business + modelling takeaways, auto-generated summary | Cell 8 |
| **8. Visualisation** | `reporting/plots.py` | Full plot suite (learning curves, heatmaps, decile charts) | Cell 9 |

### Notes
- Cells 6 & 7 map to a single `analysis/advanced.py` — advanced EDA and feature correlation are two sides of the same diagnostic pass and should share the same dataframe context.
- Cell 1 (simulation) is replaced entirely by `data/loader.py` in dev; the simulation logic can be kept as `data/simulate.py` for local testing without a live DB connection.
- `model/train.py` should enforce a **temporal train/test split** (not the random split used in the notebook).

---

## Simulated Data

| Entity | Detail |
|--------|--------|
| Users | 5,000 across 5 archetypes |
| Games | 80 games across 12 types (Baccarat, Roulette, Blackjack, Poker, etc.) |
| Vendors | Evolution, Pragmatic Play, Ezugi, Playtech, SA Gaming |
| Transactions | 194,164 over a 90-day window |
| Active users | 4,766 (avg 40.7 transactions/user) |


**User archetypes:**

| Archetype | Share | Behaviour |
|-----------|-------|-----------|
| Casual | 40% | Low-stakes game-show games (CrazyTime, Monopoly) |
| Regular | 25% | Baccarat, Roulette, DragonTiger |
| Explorer | 20% | Plays across all game types |
| Dormant | 10% | 0–1 sessions/week |
| High Roller | 5% | High-stakes Baccarat, Poker, Blackjack; evening hours |

---

## Feature Engineering

**Implicit score** — composite signal per (user, game) pair:
```
implicit_score = w1 * norm(play_count)
              + w2 * norm(total_bet)
              + w3 * norm(avg_session_duration)
              + w4 * recency_decay
```

**User features:** `device`, `country`, `preferred_game_type`

**Item features:** `gameType`, `vendor`, `isTopGame`, `isMostPopular`

**Interaction matrix:** 4,766 × 80 — density ~5%

---

## Model

| Parameter | Value |
|-----------|-------|
| Algorithm | LightFM (WARP loss) |
| Components | 64 |
| Epochs | 30 |
| Learning rate | 0.05 |
| L2 regularisation | 1e-6 (user + item) |
| Train/test split | 85/15 random (to be replaced with temporal split) |

---

## Results

| Metric | Train | Test |
|--------|-------|------|
| Precision@10 | 0.2195 | 0.0608 |
| AUC | 0.9078 | 0.7866 |

**Note:** The train/test gap indicates overfitting on simulated data — expected given the synthetic nature of interactions. Addressed by increasing regularisation and, more importantly, replacing with real data + temporal splitting.

---

## Feature Correlation with Model Score

Top features by Spearman correlation with predicted score:

| Feature | ρ | Direction |
|---------|---|-----------|
| diversity_ratio | 0.572 | + |
| unique_players | 0.424 | + |
| popularity_score | 0.242 | + |
| play_count | -0.207 | − |

The model rewards broadly appealing, cross-segment games. High per-user `play_count` correlating negatively suggests the model is not simply rewarding frequency — it's learning preference signal.

---

## Known Limitations (Current Phase)

- **Synthetic data** — all distributions are hand-crafted; real user behaviour will differ
- **Random train/test split** — must be replaced with a temporal split to avoid leakage
- **Cold-start not yet handled** — new users/games need a fallback strategy (popularity-based or content-only)
- **No A/B framework yet** — online evaluation metrics (CTR, session depth) to be defined

---

## Next Steps

1. Replace simulated data with real MongoDB → pandas load
2. Switch to temporal train/test split
3. Tune regularisation to reduce overfitting
5. Define online evaluation metrics and integrate with serving layer

---

## Stack

- Python 3.10
- LightFM, pandas, NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn
