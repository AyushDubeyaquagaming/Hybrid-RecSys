"""Nightly diagnostic plot generation for MLflow.

No Prefect @task decorator — diagnostics are best-effort reporting, not a
core training task. Called from flow.py inside a try/except guard.
"""
import os
import json
import textwrap
from collections import defaultdict

import numpy as np
import pandas as pd

from pipeline.logging_utils import get_logger


logger = get_logger(__name__)

USER_SIDE_PREFIXES = (
    "preferred_time_of_day:",
    "preferred_day_of_week:",
    "preferred_device:",
    "preferred_entry_point:",
)
ITEM_SIDE_PREFIXES = (
    "game_type:",
    "provider:",
    "popularity_bucket:",
)
MAX_ATTRIBUTION_INTERACTIONS = 20000
TOP_ATTRIBUTION_FEATURES = 8
ATTRIBUTION_STABILITY_SAMPLES = 6
ATTRIBUTION_SAMPLE_FRACTION = 0.7
PERMUTATION_TOP_FEATURES = 8
ABLATION_TOP_FEATURES = 3
EVAL_K_DEFAULT = 5
NUM_THREADS_DEFAULT = 4
LOCAL_EXPLANATION_USERS = 4
LOCAL_EXPLANATION_FEATURES = 6


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _plot_training_curve(training_history: list[dict], output_dir: str) -> None:
    """01 — Training curve: train_p5 and test_p5 over epochs."""
    import matplotlib.pyplot as plt

    if not training_history:
        logger.warning("No training history checkpoints; skipping training curve plot.")
        return

    epochs = [c["epoch"] for c in training_history]
    train_p5 = [c["train_p5"] for c in training_history]
    test_p5 = [c["test_p5"] for c in training_history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_p5, marker="o", label="Train P@5")
    ax.plot(epochs, test_p5, marker="s", label="Test P@5")
    ax.set_title("LightFM Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision@5")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_training_curve.png"), dpi=100)
    plt.close(fig)


def _plot_interaction_density(events_df: pd.DataFrame, output_dir: str) -> None:
    """02 — Sessions per user and unique games per user histograms."""
    import matplotlib.pyplot as plt

    sessions_per_user = events_df.groupby("userId").size()
    games_per_user = events_df.groupby("userId")["gameId"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sessions_per_user, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Sessions per User")
    axes[0].set_xlabel("Session count")
    axes[0].set_ylabel("Users")
    axes[0].grid(alpha=0.3)

    axes[1].hist(games_per_user, bins=20, color="darkorange", edgecolor="white")
    axes[1].set_title("Unique Games per User")
    axes[1].set_xlabel("Unique games")
    axes[1].set_ylabel("Users")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Interaction Density")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_interaction_density.png"), dpi=100)
    plt.close(fig)


def _plot_outcome_distribution(events_df: pd.DataFrame, output_dir: str) -> None:
    """03 — Outcome distribution bar chart."""
    import matplotlib.pyplot as plt

    counts = events_df["outcome"].value_counts()
    colors = {
        "net_positive": "#2ecc71",
        "net_negative": "#e74c3c",
        "break_even": "#95a5a6",
    }
    bar_colors = [colors.get(label, "#3498db") for label in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title("Outcome Distribution")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Event count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_outcome_distribution.png"), dpi=100)
    plt.close(fig)


def _plot_session_duration_coverage(events_df: pd.DataFrame, output_dir: str) -> None:
    """04 — Session duration coverage: all events and non-zero sessions."""
    import matplotlib.pyplot as plt

    durations = events_df["durationSeconds"]
    nonzero = durations[durations > 0]

    zero_pct = (durations == 0).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        np.log1p(durations), bins=40, color="mediumpurple", edgecolor="white"
    )
    axes[0].set_title(f"All Events (log1p scale)\n{zero_pct:.1f}% are zero-duration")
    axes[0].set_xlabel("log1p(durationSeconds)")
    axes[0].set_ylabel("Events")
    axes[0].grid(alpha=0.3)

    if len(nonzero) > 0:
        axes[1].hist(
            np.log1p(nonzero), bins=40, color="teal", edgecolor="white"
        )
        axes[1].set_title(f"Non-Zero Sessions Only (n={len(nonzero):,})")
        axes[1].set_xlabel("log1p(durationSeconds)")
        axes[1].set_ylabel("Events")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No non-zero sessions", ha="center", va="center")
        axes[1].set_title("Non-Zero Sessions Only")

    fig.suptitle("Session Duration Coverage")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_session_duration_coverage.png"), dpi=100)
    plt.close(fig)


def _get_id_maps(dataset) -> tuple[dict[str, int], dict[str, int], dict[int, str], dict[int, str]]:
    user_id_map, _, item_id_map, _ = dataset.mapping()
    idx_to_user = {int(index): str(user_id) for user_id, index in user_id_map.items()}
    idx_to_item = {int(index): str(item_id) for item_id, index in item_id_map.items()}
    return user_id_map, item_id_map, idx_to_user, idx_to_item


def _active_side_features_for_row(
    matrix,
    row_idx: int,
    inverse_map: dict[int, str],
    prefixes: tuple[str, ...],
) -> list[tuple[str, float, int]]:
    row = matrix.getrow(int(row_idx))
    features = []
    for feature_idx, value in zip(row.indices, row.data):
        token = inverse_map.get(int(feature_idx))
        if token and token.startswith(prefixes):
            features.append((token, float(value), int(feature_idx)))
    return features


def explain_recommendation(
    model,
    dataset_artifacts: dict,
    user_id: str,
    game_id: str,
    top_n_features: int = LOCAL_EXPLANATION_FEATURES,
    num_threads: int = NUM_THREADS_DEFAULT,
) -> dict:
    """Explain a specific user-item recommendation using LightFM side-feature contributions."""
    dataset = dataset_artifacts["dataset"]
    user_features_matrix = dataset_artifacts["user_features_matrix"]
    item_features_matrix = dataset_artifacts["item_features_matrix"]
    ife = dataset_artifacts.get("ife", pd.DataFrame())

    user_id_map, item_id_map, _, _ = _get_id_maps(dataset)
    if user_id not in user_id_map:
        raise KeyError(f"Unknown user_id: {user_id}")
    if game_id not in item_id_map:
        raise KeyError(f"Unknown game_id: {game_id}")

    user_idx = int(user_id_map[user_id])
    item_idx = int(item_id_map[game_id])
    inverse_user_feature_map, inverse_item_feature_map = _get_side_feature_maps(dataset)

    score = float(
        model.predict(
            user_ids=np.array([user_idx]),
            item_ids=np.array([item_idx]),
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=num_threads,
        )[0]
    )

    _, user_representations = model.get_user_representations(features=user_features_matrix)
    _, item_representations = model.get_item_representations(features=item_features_matrix)
    user_feature_biases = getattr(model, "user_biases", np.zeros(model.user_embeddings.shape[0]))
    item_feature_biases = getattr(model, "item_biases", np.zeros(model.item_embeddings.shape[0]))

    item_representation = item_representations[item_idx]
    user_representation = user_representations[user_idx]

    contributions = []
    for token, value, feature_idx in _active_side_features_for_row(
        user_features_matrix,
        user_idx,
        inverse_user_feature_map,
        USER_SIDE_PREFIXES,
    ):
        contribution = value * (
            user_feature_biases[feature_idx]
            + float(np.dot(model.user_embeddings[feature_idx], item_representation))
        )
        contributions.append({"side": "user", "feature": token, "contribution": contribution})

    for token, value, feature_idx in _active_side_features_for_row(
        item_features_matrix,
        item_idx,
        inverse_item_feature_map,
        ITEM_SIDE_PREFIXES,
    ):
        contribution = value * (
            item_feature_biases[feature_idx]
            + float(np.dot(user_representation, model.item_embeddings[feature_idx]))
        )
        contributions.append({"side": "item", "feature": token, "contribution": contribution})

    contributions_df = pd.DataFrame(contributions)
    metadata_total = float(contributions_df["contribution"].sum()) if not contributions_df.empty else 0.0
    residual = score - metadata_total

    if not contributions_df.empty:
        contributions_df["abs_contribution"] = contributions_df["contribution"].abs()
        top_df = contributions_df.nlargest(top_n_features, "abs_contribution").copy()
        top_df = top_df.sort_values("contribution")
    else:
        top_df = pd.DataFrame(columns=["side", "feature", "contribution", "abs_contribution"])

    item_meta = ife.loc[ife["gameId"].astype(str) == str(game_id)] if not ife.empty else pd.DataFrame()
    game_type = item_meta["game_type"].iat[0] if not item_meta.empty and "game_type" in item_meta.columns else "unknown"
    provider = item_meta["provider"].iat[0] if not item_meta.empty and "provider" in item_meta.columns else "unknown"

    return {
        "user_id": str(user_id),
        "game_id": str(game_id),
        "score": score,
        "metadata_contribution_total": metadata_total,
        "residual": residual,
        "game_type": str(game_type),
        "provider": str(provider),
        "top_contributions": top_df,
        "all_contributions": contributions_df,
    }


def _sample_local_explanations(
    model,
    dataset_artifacts: dict,
    num_users: int = LOCAL_EXPLANATION_USERS,
    top_n_features: int = LOCAL_EXPLANATION_FEATURES,
    num_threads: int = NUM_THREADS_DEFAULT,
) -> list[dict]:
    dataset = dataset_artifacts["dataset"]
    interactions = dataset_artifacts["interactions"].tocsr()
    user_features_matrix = dataset_artifacts["user_features_matrix"]
    item_features_matrix = dataset_artifacts["item_features_matrix"]

    _, _, idx_to_user, idx_to_item = _get_id_maps(dataset)
    user_interaction_counts = np.asarray(interactions.getnnz(axis=1)).ravel()
    ranked_user_indices = np.argsort(-user_interaction_counts)

    explanations = []
    used_game_ids = set()
    for user_idx in ranked_user_indices:
        user_idx = int(user_idx)
        scores = model.predict(
            user_ids=np.repeat(user_idx, interactions.shape[1]),
            item_ids=np.arange(interactions.shape[1]),
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=num_threads,
        )
        known_items = interactions[user_idx].indices
        scores[known_items] = -np.inf
        candidate_ids = np.flatnonzero(np.isfinite(scores))
        if len(candidate_ids) == 0:
            continue

        ordered_candidates = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]
        best_item_idx = None
        for candidate_idx in ordered_candidates:
            candidate_game_id = idx_to_item[int(candidate_idx)]
            if candidate_game_id not in used_game_ids:
                best_item_idx = int(candidate_idx)
                used_game_ids.add(candidate_game_id)
                break

        if best_item_idx is None:
            # Fall back to the top-ranked item if all remaining candidates are duplicates.
            best_item_idx = int(ordered_candidates[0])

        explanations.append(
            explain_recommendation(
                model,
                dataset_artifacts,
                user_id=idx_to_user[user_idx],
                game_id=idx_to_item[best_item_idx],
                top_n_features=top_n_features,
                num_threads=num_threads,
            )
        )
        if len(explanations) >= num_users:
            break

    return explanations


def _plot_local_recommendation_explanations(
    model,
    dataset_artifacts: dict,
    output_dir: str,
    num_threads: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    explanations = _sample_local_explanations(
        model,
        dataset_artifacts,
        num_users=LOCAL_EXPLANATION_USERS,
        top_n_features=LOCAL_EXPLANATION_FEATURES,
        num_threads=num_threads,
    )

    csv_records = []
    for explanation in explanations:
        for row in explanation["all_contributions"].itertuples(index=False):
            csv_records.append(
                {
                    "user_id": explanation["user_id"],
                    "game_id": explanation["game_id"],
                    "game_type": explanation["game_type"],
                    "provider": explanation["provider"],
                    "score": explanation["score"],
                    "metadata_contribution_total": explanation["metadata_contribution_total"],
                    "residual": explanation["residual"],
                    "side": row.side,
                    "feature": row.feature,
                    "contribution": row.contribution,
                }
            )
    pd.DataFrame(csv_records).to_csv(
        os.path.join(output_dir, "04_local_recommendation_explanations.csv"),
        index=False,
    )

    sns.set_theme(style="whitegrid", context="notebook")
    n_panels = max(len(explanations), 1)
    n_cols = 1 if n_panels == 1 else 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.8 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    all_values = []
    for explanation in explanations:
        if not explanation["top_contributions"].empty:
            all_values.extend(explanation["top_contributions"]["contribution"].tolist())
    x_limit = max(float(np.max(np.abs(all_values))) * 1.2, 0.05) if all_values else 0.05

    for ax, explanation in zip(axes, explanations):
        plot_df = explanation["top_contributions"].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No explanation data", ha="center", va="center")
            ax.set_axis_off()
            continue
        plot_df["display_label"] = plot_df["feature"].map(_display_feature_label)
        colors = np.where(plot_df["contribution"] >= 0, "#1b9e77", "#d95f02")
        y_positions = np.arange(len(plot_df))
        ax.barh(y_positions, plot_df["contribution"], color=colors, edgecolor="white", height=0.68)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_df["display_label"], fontsize=9)
        ax.axvline(0, color="#374151", linewidth=1.1, linestyle="--")
        ax.set_xlim(-x_limit, x_limit)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(
            f"user={explanation['user_id']} -> game={explanation['game_id']}\n"
            f"score={explanation['score']:.3f} | type={explanation['game_type']} | provider={explanation['provider']}",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        for y_pos, value in zip(y_positions, plot_df["contribution"]):
            offset = max(x_limit * 0.03, 0.006)
            ax.text(
                value + offset if value >= 0 else value - offset,
                y_pos,
                f"{value:.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
                color="#111827",
            )
        sns.despine(ax=ax, left=False, bottom=False)

    for ax in axes[len(explanations):]:
        fig.delaxes(ax)

    fig.suptitle(
        "Local Recommendation Explanations",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel("Per-feature signed contribution to recommendation score", fontsize=11)
    fig.savefig(
        os.path.join(output_dir, "04_local_recommendation_explanations.png"),
        dpi=140,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_game_popularity_coverage(
    ife: pd.DataFrame, output_dir: str, max_games: int
) -> None:
    """05 — Top-N games by popularity_score + popularity bucket counts."""
    import matplotlib.pyplot as plt

    top_games = ife.nlargest(max_games, "popularity_score")[
        ["gameId", "popularity_score"]
    ].copy()
    top_games["gameId"] = top_games["gameId"].astype(str).str[:20]  # truncate long IDs

    has_buckets = "popularity_bucket" in ife.columns

    fig, axes = plt.subplots(1, 2 if has_buckets else 1, figsize=(14 if has_buckets else 8, 5))
    if not has_buckets:
        axes = [axes]

    axes[0].barh(
        top_games["gameId"][::-1],
        top_games["popularity_score"][::-1],
        color="royalblue",
        edgecolor="white",
    )
    axes[0].set_title(f"Top {max_games} Games by Popularity Score")
    axes[0].set_xlabel("popularity_score (log1p sessions)")
    axes[0].grid(axis="x", alpha=0.3)

    if has_buckets:
        bucket_counts = ife["popularity_bucket"].value_counts()
        bucket_order = [b for b in ["cold", "warm", "hot", "blockbuster"] if b in bucket_counts.index]
        bucket_counts = bucket_counts.reindex(bucket_order)
        bucket_colors = ["#aed6f1", "#f9e79f", "#f0b27a", "#e74c3c"]
        axes[1].bar(
            bucket_counts.index,
            bucket_counts.values,
            color=bucket_colors[: len(bucket_counts)],
            edgecolor="white",
        )
        axes[1].bar_label(axes[1].containers[0], fmt="%d", padding=3)
        axes[1].set_title("Popularity Bucket Distribution")
        axes[1].set_xlabel("Bucket")
        axes[1].set_ylabel("Games")
        axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Game Catalog Coverage  (total games: {len(ife)})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_game_popularity_coverage.png"), dpi=100)
    plt.close(fig)


def _wrap_feature_label(label: str, width: int = 14) -> str:
    return "\n".join(textwrap.wrap(label.replace("_", " "), width=width))


def _correlation_columns(df: pd.DataFrame, preferred_columns: list[str]) -> list[str]:
    available = [col for col in preferred_columns if col in df.columns]
    usable = []
    for col in available:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.nunique() > 1:
            usable.append(col)
    return usable


def _plot_correlation_heatmap(ax, corr_df: pd.DataFrame, title: str):
    plot_df = corr_df.copy()
    mask = np.triu(np.ones_like(plot_df, dtype=bool), k=1)
    plot_df = plot_df.mask(mask)

    image = ax.imshow(
        plot_df.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(
        [_wrap_feature_label(col) for col in plot_df.columns],
        rotation=0,
        ha="center",
        fontsize=8,
    )
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels([_wrap_feature_label(idx) for idx in plot_df.index], fontsize=8)

    for row_idx in range(plot_df.shape[0]):
        for col_idx in range(plot_df.shape[1]):
            value = plot_df.iat[row_idx, col_idx]
            if pd.isna(value):
                continue
            text_color = "white" if abs(value) >= 0.45 else "#1f2933"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
                fontweight="semibold",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def _format_token_label(token: str) -> str:
    replacements = {
        "preferred_time_of_day:": "time=",
        "preferred_day_of_week:": "day=",
        "preferred_device:": "device=",
        "preferred_entry_point:": "entry=",
        "game_type:": "game_type=",
        "provider:": "provider=",
        "popularity_bucket:": "pop_bucket=",
    }
    for prefix, replacement in replacements.items():
        if token.startswith(prefix):
            return token.replace(prefix, replacement, 1)
    return token


def _display_feature_label(token: str, max_width: int = 18, max_length: int = 36) -> str:
    label = _format_token_label(token)
    if len(label) > max_length:
        label = f"{label[: max_length - 1]}..."
    return "\n".join(textwrap.wrap(label, width=max_width))


def _extract_side_features(matrix, row_indices: np.ndarray, inverse_map: dict[int, str], prefixes: tuple[str, ...]) -> dict[int, list[tuple[str, float, int]]]:
    feature_cache = {}
    for row_idx in np.unique(row_indices):
        row = matrix.getrow(int(row_idx))
        active = []
        for feature_idx, value in zip(row.indices, row.data):
            token = inverse_map.get(int(feature_idx))
            if token and token.startswith(prefixes):
                active.append((token, float(value), int(feature_idx)))
        feature_cache[int(row_idx)] = active
    return feature_cache


def _get_side_feature_maps(dataset) -> tuple[dict[int, str], dict[int, str]]:
    _, user_feature_map, _, item_feature_map = dataset.mapping()
    inverse_user_feature_map = {int(index): token for token, index in user_feature_map.items()}
    inverse_item_feature_map = {int(index): token for token, index in item_feature_map.items()}
    return inverse_user_feature_map, inverse_item_feature_map


def _sample_interaction_positions(train_weights, sample_fraction: float, random_state: int) -> tuple[np.ndarray, int]:
    train_weights_coo = train_weights.tocoo()
    n_interactions = train_weights_coo.nnz
    if n_interactions == 0:
        return np.array([], dtype=int), 0

    sample_size = min(
        n_interactions,
        max(1, int(np.ceil(n_interactions * sample_fraction))),
        MAX_ATTRIBUTION_INTERACTIONS,
    )
    if sample_size >= n_interactions:
        return np.arange(n_interactions), n_interactions

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_interactions, size=sample_size, replace=False)), n_interactions


def _compute_sample_feature_attribution(
    model,
    dataset_artifacts: dict,
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int]:
    dataset = dataset_artifacts.get("dataset")
    user_features_matrix = dataset_artifacts.get("user_features_matrix")
    item_features_matrix = dataset_artifacts.get("item_features_matrix")
    train_weights = dataset_artifacts.get("train_weights")

    if dataset is None or user_features_matrix is None or item_features_matrix is None or train_weights is None:
        return pd.DataFrame(columns=["side", "feature", "contribution"]), 0

    inverse_user_feature_map, inverse_item_feature_map = _get_side_feature_maps(dataset)

    train_weights_coo = train_weights.tocoo()
    sampled_positions, _ = _sample_interaction_positions(
        train_weights,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )
    if len(sampled_positions) == 0:
        return pd.DataFrame(columns=["side", "feature", "contribution"]), 0

    rows = train_weights_coo.row[sampled_positions]
    cols = train_weights_coo.col[sampled_positions]
    weights = train_weights_coo.data[sampled_positions]

    _, user_representations = model.get_user_representations(features=user_features_matrix)
    _, item_representations = model.get_item_representations(features=item_features_matrix)
    user_feature_biases = getattr(model, "user_biases", np.zeros(model.user_embeddings.shape[0]))
    item_feature_biases = getattr(model, "item_biases", np.zeros(model.item_embeddings.shape[0]))

    user_active_features = _extract_side_features(
        user_features_matrix,
        rows,
        inverse_user_feature_map,
        USER_SIDE_PREFIXES,
    )
    item_active_features = _extract_side_features(
        item_features_matrix,
        cols,
        inverse_item_feature_map,
        ITEM_SIDE_PREFIXES,
    )

    contribution_totals = defaultdict(float)
    contribution_weights = defaultdict(float)

    for user_idx, item_idx, interaction_weight in zip(rows, cols, weights):
        item_representation = item_representations[int(item_idx)]
        user_representation = user_representations[int(user_idx)]
        sample_weight = float(interaction_weight)

        for token, value, feature_idx in user_active_features.get(int(user_idx), []):
            contribution = sample_weight * value * (
                user_feature_biases[feature_idx]
                + float(np.dot(model.user_embeddings[feature_idx], item_representation))
            )
            contribution_totals[("user", token)] += contribution
            contribution_weights[("user", token)] += sample_weight

        for token, value, feature_idx in item_active_features.get(int(item_idx), []):
            contribution = sample_weight * value * (
                item_feature_biases[feature_idx]
                + float(np.dot(user_representation, model.item_embeddings[feature_idx]))
            )
            contribution_totals[("item", token)] += contribution
            contribution_weights[("item", token)] += sample_weight

    records = []
    for (side, token), total in contribution_totals.items():
        records.append(
            {
                "side": side,
                "feature": token,
                "contribution": total / max(contribution_weights[(side, token)], 1e-9),
            }
        )

    return pd.DataFrame.from_records(records), len(sampled_positions)


def _top_features_by_abs(feature_summary: pd.DataFrame, side: str | None = None, top_n: int = TOP_ATTRIBUTION_FEATURES) -> pd.DataFrame:
    working = feature_summary.copy()
    if side is not None:
        working = working[working["side"] == side].copy()
    if working.empty:
        return working
    working["abs_contribution"] = working["mean_contribution"].abs()
    selected = working.nlargest(top_n, "abs_contribution").copy()
    return selected.sort_values("abs_contribution", ascending=True)


def _plot_contribution_panel(ax, panel_df: pd.DataFrame, title: str, x_limit: float) -> None:
    import seaborn as sns

    if panel_df.empty:
        ax.text(0.5, 0.5, "No attribution data", ha="center", va="center")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_axis_off()
        return

    plot_df = panel_df.copy()
    plot_df["display_label"] = plot_df["feature"].map(_display_feature_label)
    plot_df["color"] = np.where(plot_df["mean_contribution"] >= 0, "#1b9e77", "#d95f02")
    y_positions = np.arange(len(plot_df))

    ax.barh(
        y_positions,
        plot_df["mean_contribution"],
        color=plot_df["color"],
        edgecolor="white",
        linewidth=0.8,
        height=0.68,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["display_label"], fontsize=9)
    ax.axvline(0, color="#374151", linewidth=1.2, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(-x_limit, x_limit)
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="y", pad=6)

    value_offset = max(x_limit * 0.03, 0.006)
    for y_pos, value in zip(y_positions, plot_df["mean_contribution"]):
        label_x = value + value_offset if value >= 0 else value - value_offset
        ax.text(
            label_x,
            y_pos,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color="#111827",
        )

    sns.despine(ax=ax, left=False, bottom=False)


def plot_feature_attribution(feature_summary: pd.DataFrame, output_path: str, top_n: int = TOP_ATTRIBUTION_FEATURES) -> None:
    """Create a publication-quality LightFM attribution chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")

    user_top = _top_features_by_abs(feature_summary, side="user", top_n=top_n)
    item_top = _top_features_by_abs(feature_summary, side="item", top_n=top_n)
    combined_top = _top_features_by_abs(feature_summary, side=None, top_n=top_n)

    comparable_values = pd.concat(
        [user_top["mean_contribution"], item_top["mean_contribution"]],
        ignore_index=True,
    ).to_numpy()
    x_limit = max(float(np.max(np.abs(comparable_values))) * 1.2, 0.05) if len(comparable_values) else 0.05
    combined_limit = max(float(combined_top["mean_contribution"].abs().max()) * 1.2, x_limit) if len(combined_top) else x_limit

    fig = plt.figure(figsize=(21, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.95])
    ax_user = fig.add_subplot(grid[0, 0])
    ax_item = fig.add_subplot(grid[0, 1])
    ax_combined = fig.add_subplot(grid[0, 2])

    _plot_contribution_panel(ax_user, user_top, "User Feature Contributions", x_limit=x_limit)
    _plot_contribution_panel(ax_item, item_top, "Item Feature Contributions", x_limit=x_limit)
    _plot_contribution_panel(ax_combined, combined_top, "Top Combined Features", x_limit=combined_limit)

    fig.suptitle(
        "LightFM Feature Attribution",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.965,
        "Green features lift the score; red features suppress it. User and item panels share the same x-scale for direct comparison.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.supxlabel("Mean signed contribution", fontsize=11)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def compute_stability(
    model,
    dataset_artifacts: dict,
    n_samples: int = ATTRIBUTION_STABILITY_SAMPLES,
    sample_fraction: float = ATTRIBUTION_SAMPLE_FRACTION,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Bootstrap attribution estimates to measure variance and sign consistency."""
    sample_frames = []
    sampled_counts = []
    for sample_idx in range(n_samples):
        sample_df, sampled_count = _compute_sample_feature_attribution(
            model,
            dataset_artifacts,
            sample_fraction=sample_fraction,
            random_state=random_state + sample_idx,
        )
        if sample_df.empty:
            continue
        sample_df = sample_df.copy()
        sample_df["sample_id"] = sample_idx
        sample_frames.append(sample_df)
        sampled_counts.append(sampled_count)

    if not sample_frames:
        return pd.DataFrame(
            columns=[
                "side",
                "feature",
                "mean_contribution",
                "std_deviation",
                "stability_score",
                "sign_consistency",
                "samples_observed",
            ]
        ), 0

    sample_results = pd.concat(sample_frames, ignore_index=True)
    grouped = sample_results.groupby(["side", "feature"])["contribution"]
    summary = grouped.agg(
        mean_contribution="mean",
        std_deviation=lambda s: float(np.std(s, ddof=0)),
        samples_observed="count",
    ).reset_index()
    summary["stability_score"] = summary["mean_contribution"].abs() / (
        summary["mean_contribution"].abs() + summary["std_deviation"] + 1e-9
    )

    sign_consistency = []
    for _, row in summary.iterrows():
        values = sample_results.loc[
            (sample_results["side"] == row["side"]) & (sample_results["feature"] == row["feature"]),
            "contribution",
        ]
        non_zero = values[np.abs(values) > 1e-9]
        if len(non_zero) == 0:
            sign_consistency.append(1.0)
            continue
        reference_sign = np.sign(row["mean_contribution"]) if abs(row["mean_contribution"]) > 1e-9 else 0
        if reference_sign == 0:
            sign_consistency.append(1.0)
            continue
        sign_consistency.append(float((np.sign(non_zero) == reference_sign).mean()))

    summary["sign_consistency"] = sign_consistency
    summary = summary.sort_values("mean_contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return summary, int(np.mean(sampled_counts)) if sampled_counts else 0


def _evaluate_feature_matrices(
    model,
    dataset_artifacts: dict,
    user_features_matrix,
    item_features_matrix,
    eval_k: int,
    num_threads: int,
) -> dict:
    from lightfm.evaluation import auc_score, precision_at_k

    train_interactions = dataset_artifacts["train_interactions"]
    test_interactions = dataset_artifacts["test_interactions"]
    return {
        "precision_at_k": float(
            precision_at_k(
                model,
                test_interactions,
                train_interactions=train_interactions,
                user_features=user_features_matrix,
                item_features=item_features_matrix,
                k=eval_k,
                num_threads=num_threads,
            ).mean()
        ),
        "auc": float(
            auc_score(
                model,
                test_interactions,
                train_interactions=train_interactions,
                user_features=user_features_matrix,
                item_features=item_features_matrix,
                num_threads=num_threads,
            ).mean()
        ),
    }


def _permute_sparse_feature_column(matrix, column_idx: int, random_state: int):
    permuted = matrix.tolil(copy=True)
    column_values = np.asarray(permuted[:, column_idx].toarray()).ravel()
    rng = np.random.default_rng(random_state)
    rng.shuffle(column_values)
    permuted[:, column_idx] = column_values.reshape(-1, 1)
    return permuted.tocsr()


def run_permutation_test(
    model,
    dataset_artifacts: dict,
    feature_summary: pd.DataFrame,
    top_n: int = PERMUTATION_TOP_FEATURES,
    eval_k: int = EVAL_K_DEFAULT,
    num_threads: int = NUM_THREADS_DEFAULT,
    random_state: int = 42,
) -> pd.DataFrame:
    """Shuffle top features individually and measure degradation in ranking quality."""
    dataset = dataset_artifacts.get("dataset")
    if dataset is None or feature_summary.empty:
        return pd.DataFrame(columns=["side", "feature", "delta_precision_at_k", "delta_auc"])

    base_metrics = _evaluate_feature_matrices(
        model,
        dataset_artifacts,
        dataset_artifacts["user_features_matrix"],
        dataset_artifacts["item_features_matrix"],
        eval_k=eval_k,
        num_threads=num_threads,
    )

    _, user_feature_map, _, item_feature_map = dataset.mapping()
    top_features = _top_features_by_abs(feature_summary, top_n=top_n)
    records = []
    for idx, row in enumerate(top_features.itertuples(index=False)):
        if row.side == "user":
            feature_idx = user_feature_map.get(row.feature)
            if feature_idx is None:
                continue
            user_matrix = _permute_sparse_feature_column(
                dataset_artifacts["user_features_matrix"],
                feature_idx,
                random_state=random_state + idx,
            )
            item_matrix = dataset_artifacts["item_features_matrix"]
        else:
            feature_idx = item_feature_map.get(row.feature)
            if feature_idx is None:
                continue
            user_matrix = dataset_artifacts["user_features_matrix"]
            item_matrix = _permute_sparse_feature_column(
                dataset_artifacts["item_features_matrix"],
                feature_idx,
                random_state=random_state + idx,
            )

        permuted_metrics = _evaluate_feature_matrices(
            model,
            dataset_artifacts,
            user_matrix,
            item_matrix,
            eval_k=eval_k,
            num_threads=num_threads,
        )
        records.append(
            {
                "side": row.side,
                "feature": row.feature,
                "delta_precision_at_k": base_metrics["precision_at_k"] - permuted_metrics["precision_at_k"],
                "delta_auc": base_metrics["auc"] - permuted_metrics["auc"],
            }
        )

    return pd.DataFrame.from_records(records)


def _run_ablation_check(
    model,
    dataset_artifacts: dict,
    feature_summary: pd.DataFrame,
    top_k: int = ABLATION_TOP_FEATURES,
    eval_k: int = EVAL_K_DEFAULT,
    num_threads: int = NUM_THREADS_DEFAULT,
) -> dict:
    dataset = dataset_artifacts.get("dataset")
    if dataset is None or feature_summary.empty:
        return {}

    _, user_feature_map, _, item_feature_map = dataset.mapping()
    base_metrics = _evaluate_feature_matrices(
        model,
        dataset_artifacts,
        dataset_artifacts["user_features_matrix"],
        dataset_artifacts["item_features_matrix"],
        eval_k=eval_k,
        num_threads=num_threads,
    )

    ablated_user_matrix = dataset_artifacts["user_features_matrix"].tolil(copy=True)
    ablated_item_matrix = dataset_artifacts["item_features_matrix"].tolil(copy=True)
    selected_features = _top_features_by_abs(feature_summary, top_n=top_k)

    for row in selected_features.itertuples(index=False):
        if row.side == "user" and row.feature in user_feature_map:
            ablated_user_matrix[:, user_feature_map[row.feature]] = 0.0
        if row.side == "item" and row.feature in item_feature_map:
            ablated_item_matrix[:, item_feature_map[row.feature]] = 0.0

    ablated_metrics = _evaluate_feature_matrices(
        model,
        dataset_artifacts,
        ablated_user_matrix.tocsr(),
        ablated_item_matrix.tocsr(),
        eval_k=eval_k,
        num_threads=num_threads,
    )

    return {
        "top_k_ablated": int(top_k),
        "features": selected_features[["side", "feature", "mean_contribution"]].to_dict(orient="records"),
        "baseline_precision_at_k": base_metrics["precision_at_k"],
        "baseline_auc": base_metrics["auc"],
        "ablated_precision_at_k": ablated_metrics["precision_at_k"],
        "ablated_auc": ablated_metrics["auc"],
        "precision_at_k_drop": base_metrics["precision_at_k"] - ablated_metrics["precision_at_k"],
        "auc_drop": base_metrics["auc"] - ablated_metrics["auc"],
    }


def _plot_validation_panels(feature_summary: pd.DataFrame, permutation_df: pd.DataFrame, output_path: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")

    top_summary = _top_features_by_abs(feature_summary, top_n=10).copy()
    top_summary = top_summary.sort_values("mean_contribution", key=lambda s: s.abs(), ascending=True)
    top_summary["display_label"] = top_summary["feature"].map(_display_feature_label)

    merged_perm = permutation_df.merge(
        feature_summary[["side", "feature", "mean_contribution"]],
        on=["side", "feature"],
        how="left",
    ) if not permutation_df.empty else pd.DataFrame(columns=["feature", "delta_precision_at_k", "delta_auc", "mean_contribution"])
    if not merged_perm.empty:
        merged_perm = merged_perm.sort_values("delta_precision_at_k", ascending=True)
        merged_perm["display_label"] = merged_perm["feature"].map(_display_feature_label)

    fig = plt.figure(figsize=(18, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    if len(top_summary) > 0:
        y_positions = np.arange(len(top_summary))
        colors = np.where(top_summary["mean_contribution"] >= 0, "#1b9e77", "#d95f02")
        ax_left.barh(y_positions, top_summary["mean_contribution"], color=colors, edgecolor="white")
        ax_left.errorbar(
            top_summary["mean_contribution"],
            y_positions,
            xerr=top_summary["std_deviation"],
            fmt="none",
            ecolor="#374151",
            alpha=0.7,
            capsize=3,
        )
        ax_left.set_yticks(y_positions)
        ax_left.set_yticklabels(top_summary["display_label"], fontsize=9)
        ax_left.axvline(0, color="#374151", linestyle="--", linewidth=1)
        ax_left.set_title("Attribution Stability (mean ± std)", fontsize=12, fontweight="bold")
        ax_left.set_xlabel("Mean contribution")
        ax_left.grid(axis="x", alpha=0.2)
    else:
        ax_left.text(0.5, 0.5, "No stability data", ha="center", va="center")
        ax_left.set_axis_off()

    if len(merged_perm) > 0:
        y_positions = np.arange(len(merged_perm))
        ax_right.barh(y_positions, merged_perm["delta_precision_at_k"], color="#4c78a8", edgecolor="white")
        ax_right.set_yticks(y_positions)
        ax_right.set_yticklabels(merged_perm["display_label"], fontsize=9)
        ax_right.axvline(0, color="#374151", linestyle="--", linewidth=1)
        ax_right.set_title("Permutation Impact on Precision@K", fontsize=12, fontweight="bold")
        ax_right.set_xlabel("Precision@K drop after shuffling")
        ax_right.grid(axis="x", alpha=0.2)
    else:
        ax_right.text(0.5, 0.5, "No permutation data", ha="center", va="center")
        ax_right.set_axis_off()

    sns.despine(fig=fig)
    fig.suptitle("Feature Attribution Validation", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_correlation(dataset_artifacts: dict, output_dir: str) -> None:
    """06 — Spearman correlation heatmaps for user, item, and interaction features."""
    import matplotlib.pyplot as plt

    user_df = dataset_artifacts.get("ufe", pd.DataFrame())
    item_df = dataset_artifacts.get("ife", pd.DataFrame())
    interaction_df = dataset_artifacts.get("train_user_game_df", pd.DataFrame())

    plot_specs = [
        (
            user_df,
            [
                "total_sessions",
                "unique_games",
                "unique_providers",
                "unique_game_types",
                "avg_duration_sec",
                "avg_rounds",
                "quick_exit_rate",
                "return_10m_rate",
                "positive_outcome_rate",
                "avg_engagement_intensity",
                "recency_days",
            ],
            "User Features",
        ),
        (
            item_df,
            [
                "game_sessions",
                "unique_users",
                "avg_duration_sec",
                "avg_rounds",
                "quick_exit_rate",
                "return_10m_rate",
                "positive_outcome_rate",
                "popularity_score",
            ],
            "Item Features",
        ),
        (
            interaction_df,
            [
                "interaction_count",
                "avg_duration_sec",
                "avg_rounds",
                "positive_outcome_rate",
                "return_10m_rate",
                "avg_engagement_intensity",
                "recency_days",
                "implicit_score",
            ],
            "Interaction Features vs Implicit Score",
        ),
    ]

    fig = plt.figure(figsize=(21, 8), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    cax = fig.add_subplot(grid[0, 3])
    colorbar_image = None

    for ax, (source_df, preferred_columns, title) in zip(axes, plot_specs):
        usable_columns = _correlation_columns(source_df, preferred_columns)
        if len(usable_columns) < 2:
            ax.text(0.5, 0.5, "Not enough variable features", ha="center", va="center")
            ax.set_title(title)
            ax.set_axis_off()
            continue

        corr_df = source_df[usable_columns].corr(method="spearman")
        colorbar_image = _plot_correlation_heatmap(ax, corr_df, title)

    fig.suptitle(
        "Feature Association Diagnostics (Spearman Correlation)",
        fontsize=15,
        fontweight="bold",
    )
    if colorbar_image is not None:
        colorbar = fig.colorbar(colorbar_image, cax=cax)
        colorbar.set_label("Spearman correlation", rotation=90, labelpad=12)
    else:
        cax.set_axis_off()
    fig.savefig(os.path.join(output_dir, "06_feature_correlation.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_artifact(output_dir: str, filename: str, payload: dict) -> None:
    with open(os.path.join(output_dir, filename), "w") as fp:
        json.dump(payload, fp, indent=2, default=_json_default)


def _known_rate(series: pd.Series, unknown_values: set[str] | None = None) -> float:
    if len(series) == 0:
        return 0.0
    if unknown_values is None:
        unknown_values = {"unknown", "nan", "none", "null", ""}
    normalized = series.astype(str).str.strip().str.lower()
    return float((~normalized.isin(unknown_values)).mean())


def _session_tier(session_count: float) -> str:
    if session_count <= 5:
        return "1_5"
    if session_count <= 20:
        return "6_20"
    return "21_plus"


def _preference_rate_by_group(df: pd.DataFrame, group_col: str, pref_col: str, pref_value: str) -> pd.Series:
    return (
        df.assign(_flag=(df[pref_col].astype(str) == pref_value).astype(float))
        .groupby(group_col)["_flag"]
        .mean()
    )


def _write_dataset_overview(events_df: pd.DataFrame, dataset_artifacts: dict, output_dir: str) -> None:
    interactions = dataset_artifacts["interactions"]
    train_interactions = dataset_artifacts["train_interactions"]
    test_interactions = dataset_artifacts["test_interactions"]
    user_feature_vocab = dataset_artifacts.get("user_feature_vocab", [])
    item_feature_vocab = dataset_artifacts.get("item_feature_vocab", [])

    sessions_per_user = events_df.groupby("userId").size() if not events_df.empty else pd.Series(dtype=float)
    games_per_user = (
        events_df.groupby("userId")["gameId"].nunique() if not events_df.empty else pd.Series(dtype=float)
    )
    implicit_scores = dataset_artifacts.get("train_user_game_df", pd.DataFrame()).get(
        "implicit_score", pd.Series(dtype=float)
    )

    overview = {
        "n_events": int(len(events_df)),
        "n_active_users": int(events_df["userId"].nunique()) if not events_df.empty else 0,
        "n_active_games": int(events_df["gameId"].nunique()) if not events_df.empty else 0,
        "train_nnz": int(train_interactions.nnz),
        "test_nnz": int(test_interactions.nnz),
        "full_nnz": int(interactions.nnz),
        "interaction_density_pct": float(
            interactions.nnz / max(interactions.shape[0] * interactions.shape[1], 1) * 100.0
        ),
        "user_feature_vocab_size": int(len(user_feature_vocab)),
        "item_feature_vocab_size": int(len(item_feature_vocab)),
        "avg_sessions_per_user": float(sessions_per_user.mean()) if len(sessions_per_user) else 0.0,
        "median_sessions_per_user": float(sessions_per_user.median()) if len(sessions_per_user) else 0.0,
        "avg_unique_games_per_user": float(games_per_user.mean()) if len(games_per_user) else 0.0,
        "median_unique_games_per_user": float(games_per_user.median()) if len(games_per_user) else 0.0,
        "avg_implicit_score": float(implicit_scores.mean()) if len(implicit_scores) else 0.0,
        "median_implicit_score": float(implicit_scores.median()) if len(implicit_scores) else 0.0,
        "timestamp_min": events_df["timestamp"].min() if not events_df.empty else None,
        "timestamp_max": events_df["timestamp"].max() if not events_df.empty else None,
    }
    _write_json_artifact(output_dir, "11_dataset_overview.json", overview)


def _write_enrichment_coverage(events_df: pd.DataFrame, output_dir: str) -> None:
    coverage = {
        "deviceType_known_rate": _known_rate(events_df["deviceType"]) if "deviceType" in events_df.columns else 0.0,
        "entryPoint_known_rate": _known_rate(events_df["entryPoint"]) if "entryPoint" in events_df.columns else 0.0,
        "provider_known_rate": _known_rate(events_df["provider"]) if "provider" in events_df.columns else 0.0,
        "gameType_known_rate": _known_rate(events_df["gameType"]) if "gameType" in events_df.columns else 0.0,
        "deviceType_counts": events_df["deviceType"].value_counts(dropna=False).to_dict() if "deviceType" in events_df.columns else {},
        "entryPoint_counts": events_df["entryPoint"].value_counts(dropna=False).to_dict() if "entryPoint" in events_df.columns else {},
        "provider_counts": events_df["provider"].value_counts(dropna=False).head(10).to_dict() if "provider" in events_df.columns else {},
        "gameType_counts": events_df["gameType"].value_counts(dropna=False).to_dict() if "gameType" in events_df.columns else {},
    }
    _write_json_artifact(output_dir, "12_enrichment_coverage.json", coverage)


def _write_user_segment_summary(dataset_artifacts: dict, output_dir: str) -> None:
    ufe = dataset_artifacts.get("ufe", pd.DataFrame()).copy()
    if ufe.empty:
        pd.DataFrame().to_csv(os.path.join(output_dir, "13_user_segment_summary.csv"), index=False)
        return

    ufe["session_tier"] = ufe["total_sessions"].map(_session_tier)
    summary = (
        ufe.groupby("session_tier")
        .agg(
            user_count=("userId", "count"),
            avg_sessions=("total_sessions", "mean"),
            median_sessions=("total_sessions", "median"),
            avg_unique_games=("unique_games", "mean"),
            avg_duration_sec=("avg_duration_sec", "mean"),
            avg_quick_exit_rate=("quick_exit_rate", "mean"),
            avg_return_10m_rate=("return_10m_rate", "mean"),
            avg_positive_outcome_rate=("positive_outcome_rate", "mean"),
        )
        .reset_index()
    )

    for device in ["mobile", "desktop", "tablet", "unknown"]:
        summary[f"pct_device_{device}"] = summary["session_tier"].map(
            _preference_rate_by_group(ufe, "session_tier", "preferred_device", device)
        )

    for time_bucket in ["morning", "afternoon", "evening", "late_night"]:
        summary[f"pct_time_{time_bucket}"] = summary["session_tier"].map(
            _preference_rate_by_group(ufe, "session_tier", "preferred_time_of_day", time_bucket)
        )

    tier_order = {"1_5": 0, "6_20": 1, "21_plus": 2}
    summary = summary.sort_values("session_tier", key=lambda s: s.map(tier_order)).reset_index(drop=True)
    summary.to_csv(os.path.join(output_dir, "13_user_segment_summary.csv"), index=False)


def _write_item_catalog_summary(dataset_artifacts: dict, output_dir: str) -> None:
    ife = dataset_artifacts.get("ife", pd.DataFrame()).copy()
    if ife.empty:
        _write_json_artifact(output_dir, "14_item_catalog_summary.json", {})
        return

    summary = {
        "total_games": int(len(ife)),
        "provider_distribution": ife["provider"].astype(str).value_counts().to_dict(),
        "game_type_distribution": ife["game_type"].astype(str).value_counts().to_dict(),
        "popularity_bucket_distribution": ife["popularity_bucket"].astype(str).value_counts().to_dict() if "popularity_bucket" in ife.columns else {},
        "avg_users_per_game": float(ife["unique_users"].mean()) if "unique_users" in ife.columns else 0.0,
        "median_users_per_game": float(ife["unique_users"].median()) if "unique_users" in ife.columns else 0.0,
        "avg_sessions_per_game": float(ife["game_sessions"].mean()) if "game_sessions" in ife.columns else 0.0,
        "median_sessions_per_game": float(ife["game_sessions"].median()) if "game_sessions" in ife.columns else 0.0,
        "cold_item_count": int((ife["popularity_bucket"].astype(str) == "cold").sum()) if "popularity_bucket" in ife.columns else 0,
    }
    _write_json_artifact(output_dir, "14_item_catalog_summary.json", summary)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_diagnostic_plots(
    dataset_artifacts: dict,
    events_df: pd.DataFrame,
    training_history: list[dict],
    output_dir: str,
    max_games_to_plot: int = 15,
    eval_k: int = EVAL_K_DEFAULT,
    num_threads: int = NUM_THREADS_DEFAULT,
    model=None,
) -> str:
    """Generate nightly diagnostic PNG plots and save them to output_dir.

    Returns:
        output_dir
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for nightly jobs

    os.makedirs(output_dir, exist_ok=True)
    ife = dataset_artifacts["ife"]

    _plot_training_curve(training_history, output_dir)
    logger.info("Diagnostic: training curve written")

    _plot_interaction_density(events_df, output_dir)
    logger.info("Diagnostic: interaction density written")

    _plot_outcome_distribution(events_df, output_dir)
    logger.info("Diagnostic: outcome distribution written")

    if model is not None:
        _plot_local_recommendation_explanations(
            model,
            dataset_artifacts,
            output_dir,
            num_threads=num_threads,
        )
        logger.info("Diagnostic: local recommendation explanations written")

    _plot_game_popularity_coverage(ife, output_dir, max_games=max_games_to_plot)
    logger.info("Diagnostic: game popularity coverage written")

    _plot_feature_correlation(dataset_artifacts, output_dir)
    logger.info("Diagnostic: feature correlation written")

    _write_dataset_overview(events_df, dataset_artifacts, output_dir)
    logger.info("Diagnostic: dataset overview written")

    _write_enrichment_coverage(events_df, output_dir)
    logger.info("Diagnostic: enrichment coverage written")

    _write_user_segment_summary(dataset_artifacts, output_dir)
    logger.info("Diagnostic: user segment summary written")

    _write_item_catalog_summary(dataset_artifacts, output_dir)
    logger.info("Diagnostic: item catalog summary written")

    if model is not None:
        feature_summary, sampled_interactions = compute_stability(
            model,
            dataset_artifacts,
            n_samples=ATTRIBUTION_STABILITY_SAMPLES,
            sample_fraction=ATTRIBUTION_SAMPLE_FRACTION,
            random_state=42,
        )
        permutation_df = run_permutation_test(
            model,
            dataset_artifacts,
            feature_summary,
            top_n=PERMUTATION_TOP_FEATURES,
            eval_k=eval_k,
            num_threads=num_threads,
            random_state=42,
        )
        ablation_summary = _run_ablation_check(
            model,
            dataset_artifacts,
            feature_summary,
            top_k=ABLATION_TOP_FEATURES,
            eval_k=eval_k,
            num_threads=num_threads,
        )

        if not feature_summary.empty:
            feature_summary = feature_summary.merge(
                permutation_df,
                on=["side", "feature"],
                how="left",
            )
        feature_summary.to_csv(
            os.path.join(output_dir, "08_feature_attribution_validation.csv"),
            index=False,
        )
        with open(os.path.join(output_dir, "10_feature_ablation_summary.json"), "w") as fp:
            json.dump(
                {
                    "sampled_interactions": sampled_interactions,
                    **ablation_summary,
                },
                fp,
                indent=2,
            )

        plot_feature_attribution(
            feature_summary,
            os.path.join(output_dir, "07_feature_attribution.png"),
            top_n=TOP_ATTRIBUTION_FEATURES,
        )
        logger.info("Diagnostic: feature attribution written")

    plots = sorted(os.listdir(output_dir))
    logger.info("Diagnostic plots generated: %s", plots)
    print(f"Diagnostic plots written to {output_dir}: {plots}")

    return output_dir
