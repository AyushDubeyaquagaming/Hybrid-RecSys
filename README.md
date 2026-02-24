# Live Casino Recommendation System
**Quantum Horizon — Phase 1 (Experimentation)**

---

## Overview

A hybrid collaborative + content-based recommendation system built with [LightFM](https://github.com/lyst/lightfm), designed to surface personalized live casino game recommendations for each user. This notebook covers the full experimentation pipeline: data simulation, EDA, feature engineering, model training, evaluation, and diagnostic analysis.

> **Status:** Proof-of-concept on simulated data. Data layer will be swapped to real MongoDB → DuckDB → pandas once the pipeline is ready.

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 0 | Installs & Imports |
| 1 | Data Simulation (Games, Users, Transactions) |
| 2 | EDA |
| 3 | Feature Engineering (Interaction Matrix + User/Item Features) |
| 4 | LightFM Model Training |
| 5 | Evaluation (Precision@K, AUC) |
| 6 | Advanced EDA — user behaviour, sparsity |
| 7 | Advanced Feature Engineering — ablation + correlation |
| 8 | Auto-interpretation (business + modelling takeaways) |
| 9 | Extended visualisation suite |

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
