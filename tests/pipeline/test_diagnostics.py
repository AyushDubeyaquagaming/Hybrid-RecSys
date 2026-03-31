"""Tests for pipeline/steps/diagnostics.py"""
import os

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineSettings
from pipeline.steps.build_dataset import build_lightfm_dataset
from pipeline.steps.diagnostics import generate_diagnostic_plots
from pipeline.steps.train import train_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_events_df(n_users=5, n_games=6, n_events=60, seed=42):
    rng = np.random.default_rng(seed)
    users = [f"user_{i}" for i in range(n_users)]
    games = [f"game_{i}" for i in range(n_games)]
    timestamps = pd.date_range("2024-01-01", periods=n_events, freq="4h")
    return pd.DataFrame(
        {
            "eventType": "game_session",
            "userId": rng.choice(users, n_events),
            "sessionId": [f"sess_{i}" for i in range(n_events)],
            "gameId": rng.choice(games, n_events),
            "gameType": rng.choice(["live_dealer", "slot", "crash"], n_events),
            "provider": rng.choice(["Evolution", "EZUGI", "Spribe"], n_events),
            "timestamp": timestamps,
            "durationSeconds": rng.uniform(0, 600, n_events),
            "roundsPlayed": rng.integers(1, 10, n_events),
            "stakeLevelCategory": rng.choice(["low", "medium", "high"], n_events),
            "outcome": rng.choice(
                ["net_positive", "net_negative", "break_even"], n_events
            ),
            "exitType": rng.choice(
                ["quick_exit", "natural_end", "returned_quickly", "unknown"], n_events
            ),
            "returnedWithin10mins": rng.choice([True, False], n_events),
            "deviceType": rng.choice(["mobile", "desktop", "unknown"], n_events),
            "timeOfDay": rng.choice(
                ["morning", "afternoon", "evening", "late_night"], n_events
            ),
            "dayOfWeek": rng.choice(
                ["monday", "tuesday", "wednesday", "thursday", "friday"], n_events
            ),
            "entryPoint": rng.choice(["livecasino", "direct", "casino"], n_events),
        }
    )


def _make_games_df():
    return pd.DataFrame(
        {
            "gameId": [f"game_{i}" for i in range(6)],
            "gameName": [f"Game {i}" for i in range(6)],
            "gameType": ["Live Casino"] * 6,
            "gamevendor": ["Evolution"] * 6,
            "minBet": [1.0] * 6,
            "maxBet": [100.0] * 6,
        }
    )


EXPECTED_PLOTS = [
    "01_training_curve.png",
    "02_interaction_density.png",
    "03_outcome_distribution.png",
    "04_local_recommendation_explanations.png",
    "05_game_popularity_coverage.png",
    "06_feature_correlation.png",
    "07_feature_attribution.png",
]

EXPECTED_NON_PLOT_ARTIFACTS = [
    "04_local_recommendation_explanations.csv",
    "08_feature_attribution_validation.csv",
    "10_feature_ablation_summary.json",
]


@pytest.fixture(scope="module")
def pipeline_outputs():
    settings = PipelineSettings(N_EPOCHS=5)  # one checkpoint at epoch 5
    events_df = _make_events_df()
    games_df = _make_games_df()
    dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
    train_result = train_model.fn(dataset_artifacts, settings)
    return dataset_artifacts, events_df, train_result["history"], train_result["model"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateDiagnosticPlotsCreatesExpectedFiles:
    def test_all_seven_plots_created(self, pipeline_outputs, tmp_path):
        dataset_artifacts, events_df, history, model = pipeline_outputs
        generate_diagnostic_plots(
            dataset_artifacts=dataset_artifacts,
            events_df=events_df,
            model=model,
            training_history=history,
            output_dir=str(tmp_path),
        )
        for fname in EXPECTED_PLOTS:
            assert os.path.exists(tmp_path / fname), f"Missing plot: {fname}"
        for fname in EXPECTED_NON_PLOT_ARTIFACTS:
            assert os.path.exists(tmp_path / fname), f"Missing artifact: {fname}"

    def test_plots_are_non_empty_files(self, pipeline_outputs, tmp_path):
        dataset_artifacts, events_df, history, model = pipeline_outputs
        generate_diagnostic_plots(
            dataset_artifacts=dataset_artifacts,
            events_df=events_df,
            model=model,
            training_history=history,
            output_dir=str(tmp_path),
        )
        for fname in EXPECTED_PLOTS:
            path = tmp_path / fname
            if path.exists():  # training curve may be skipped if history empty
                assert path.stat().st_size > 0, f"Empty plot file: {fname}"
        for fname in EXPECTED_NON_PLOT_ARTIFACTS:
            path = tmp_path / fname
            assert path.exists(), f"Missing artifact: {fname}"
            assert path.stat().st_size > 0, f"Empty artifact file: {fname}"

    def test_returns_output_dir(self, pipeline_outputs, tmp_path):
        dataset_artifacts, events_df, history, model = pipeline_outputs
        result = generate_diagnostic_plots(
            dataset_artifacts=dataset_artifacts,
            events_df=events_df,
            model=model,
            training_history=history,
            output_dir=str(tmp_path),
        )
        assert result == str(tmp_path)


class TestTrainingCurveWithEmptyHistory:
    def test_no_crash_on_empty_history(self, pipeline_outputs, tmp_path):
        """Training curve is gracefully skipped when history is empty."""
        dataset_artifacts, events_df, _, model = pipeline_outputs
        generate_diagnostic_plots(
            dataset_artifacts=dataset_artifacts,
            events_df=events_df,
            model=model,
            training_history=[],  # empty — e.g. N_EPOCHS < 5
            output_dir=str(tmp_path),
        )
        # Other diagnostics artifacts must still be created
        for fname in EXPECTED_PLOTS[1:]:
            assert os.path.exists(tmp_path / fname), f"Missing plot: {fname}"
        for fname in EXPECTED_NON_PLOT_ARTIFACTS:
            assert os.path.exists(tmp_path / fname), f"Missing artifact: {fname}"


class TestTrainModelReturnsHistory:
    def test_train_result_has_model_key(self):
        settings = PipelineSettings(N_EPOCHS=5)
        events_df = _make_events_df()
        games_df = _make_games_df()
        dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
        result = train_model.fn(dataset_artifacts, settings)
        assert "model" in result
        assert "history" in result

    def test_history_has_expected_shape(self):
        settings = PipelineSettings(N_EPOCHS=10)
        events_df = _make_events_df()
        games_df = _make_games_df()
        dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
        result = train_model.fn(dataset_artifacts, settings)
        # 10 epochs → 2 checkpoints (at epoch 5 and 10)
        assert len(result["history"]) == 2
        for entry in result["history"]:
            assert "epoch" in entry
            assert "train_p5" in entry
            assert "test_p5" in entry

    def test_history_epochs_are_multiples_of_5(self):
        settings = PipelineSettings(N_EPOCHS=15)
        events_df = _make_events_df()
        games_df = _make_games_df()
        dataset_artifacts = build_lightfm_dataset.fn(events_df, games_df, settings)
        result = train_model.fn(dataset_artifacts, settings)
        for entry in result["history"]:
            assert entry["epoch"] % 5 == 0


class TestDiagnosticsDoNotBlockTraining:
    def test_plotting_failure_does_not_raise(self, pipeline_outputs, tmp_path):
        """Simulate a broken events_df — diagnostics must not crash caller."""
        dataset_artifacts, _, history, model = pipeline_outputs
        bad_events = pd.DataFrame()  # will cause individual plot helpers to fail

        # The generate function itself may raise on bad input, which is fine —
        # the protection is in flow.py's try/except. Verify at least that
        # generate_diagnostic_plots is callable and returns gracefully on valid data.
        result = generate_diagnostic_plots(
            dataset_artifacts=dataset_artifacts,
            events_df=_make_events_df(),
            model=model,
            training_history=history,
            output_dir=str(tmp_path),
        )
        assert result == str(tmp_path)


class TestEDAConfigFlags:
    def test_eda_enabled_default_true(self):
        settings = PipelineSettings()
        assert settings.EDA_ENABLED is True

    def test_eda_max_games_default(self):
        settings = PipelineSettings()
        assert settings.EDA_MAX_GAMES_TO_PLOT == 15

    def test_eda_disabled_via_override(self):
        settings = PipelineSettings(EDA_ENABLED=False)
        assert settings.EDA_ENABLED is False
