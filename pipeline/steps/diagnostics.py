"""Nightly diagnostic plot generation for MLflow.

No Prefect @task decorator — diagnostics are best-effort reporting, not a
core training task. Called from flow.py inside a try/except guard.
"""
import os

import numpy as np
import pandas as pd

from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_diagnostic_plots(
    dataset_artifacts: dict,
    events_df: pd.DataFrame,
    training_history: list[dict],
    output_dir: str,
    max_games_to_plot: int = 15,
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

    _plot_session_duration_coverage(events_df, output_dir)
    logger.info("Diagnostic: session duration coverage written")

    _plot_game_popularity_coverage(ife, output_dir, max_games=max_games_to_plot)
    logger.info("Diagnostic: game popularity coverage written")

    plots = sorted(os.listdir(output_dir))
    logger.info("Diagnostic plots generated: %s", plots)
    print(f"Diagnostic plots written to {output_dir}: {plots}")

    return output_dir
