import os
import tempfile
from contextlib import nullcontext

from prefect import flow

from pipeline.config import PipelineSettings
from pipeline.logging_utils import get_logger
from pipeline.steps import align, build_dataset, enrich, evaluate, export, ingest, train
from pipeline.steps.diagnostics import generate_diagnostic_plots


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MLflow helpers — all MLflow calls live here so the flow body stays clean
# ---------------------------------------------------------------------------

def _mlflow_run_ctx(settings: PipelineSettings):
    """Return an active MLflow run context, or a no-op context if disabled."""
    if settings.MLFLOW_ENABLED:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        return mlflow.start_run()
    return nullcontext()


def _mlflow_log_params(settings: PipelineSettings, params: dict) -> None:
    if settings.MLFLOW_ENABLED:
        import mlflow
        mlflow.log_params(params)


def _mlflow_log_metrics(settings: PipelineSettings, metrics: dict) -> None:
    if settings.MLFLOW_ENABLED:
        import mlflow
        mlflow.log_metrics(metrics)


def _mlflow_log_artifacts(settings: PipelineSettings, artifact_path: str) -> None:
    if settings.MLFLOW_ENABLED:
        import mlflow
        mlflow.log_artifacts(artifact_path, artifact_path="model_artifacts")


def _mlflow_log_diagnostics(settings: PipelineSettings, diagnostics_dir: str) -> None:
    if settings.MLFLOW_ENABLED:
        import mlflow
        mlflow.log_artifacts(diagnostics_dir, artifact_path="diagnostic_plots")


def _mlflow_log_registry_model(settings: PipelineSettings, artifact_path: str):
    """Log a registry-compatible MLflow pyfunc model. Returns the model URI."""
    import mlflow
    import mlflow.pyfunc

    from pipeline.mlflow_pyfunc import LightFMPyFuncModel

    artifact_path = os.path.abspath(artifact_path)

    artifacts = {
        "model": os.path.join(artifact_path, "model.joblib"),
        "user_features_matrix": os.path.join(artifact_path, "user_features_matrix.joblib"),
        "item_features_matrix": os.path.join(artifact_path, "item_features_matrix.joblib"),
        "interactions": os.path.join(artifact_path, "interactions.joblib"),
        "user_id_map": os.path.join(artifact_path, "user_id_map.json"),
        "item_id_map": os.path.join(artifact_path, "item_id_map.json"),
        "game_metadata": os.path.join(artifact_path, "game_metadata.json"),
        "popularity_ranking": os.path.join(artifact_path, "popularity_ranking.json"),
    }

    mlflow.pyfunc.log_model(
        artifact_path="registry_model",
        python_model=LightFMPyFuncModel(),
        artifacts=artifacts,
        pip_requirements=[
            "lightfm==1.17",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "joblib>=1.3.0",
            "pandas>=2.0.0",
        ],
    )

    run_id = mlflow.active_run().info.run_id
    return f"runs:/{run_id}/registry_model"


def _register_model(
    settings: PipelineSettings,
    model_uri: str,
    metrics: dict,
    train_nnz: int,
) -> None:
    """Register the trained model in the MLflow model registry. Best-effort — never blocks the pipeline."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    try:
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=settings.MLFLOW_REGISTERED_MODEL_NAME,
        )

        for key, metric_key in [
            ("test_precision_at_5", "test_precision_at_k"),
            ("test_auc", "test_auc"),
        ]:
            client.set_model_version_tag(
                name=settings.MLFLOW_REGISTERED_MODEL_NAME,
                version=mv.version,
                key=key,
                value=str(metrics.get(metric_key, "N/A")),
            )

        client.set_model_version_tag(
            name=settings.MLFLOW_REGISTERED_MODEL_NAME,
            version=mv.version,
            key="train_nnz",
            value=str(train_nnz),
        )

        client.transition_model_version_stage(
            name=settings.MLFLOW_REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
        )

        logger.info(
            "Model registered: %s v%s → Staging",
            settings.MLFLOW_REGISTERED_MODEL_NAME,
            mv.version,
        )

    except Exception as exc:
        logger.warning("MLflow model registration failed (best-effort): %s", exc)


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="betblitz-recsys-training", log_prints=True)
def training_flow():
    settings = PipelineSettings()
    logger.info("Starting training flow with artifact dir=%s", settings.ARTIFACT_DIR)

    # Stage 1: Data ingestion
    transactions_df = ingest.load_transactions(settings)
    games_df = ingest.load_game_details(settings)
    users_df = ingest.load_users(settings)
    transactions_df = ingest.clean_and_merge(transactions_df, games_df, settings)

    # Stage 2: Enrichment
    transactions_df = enrich.enrich_sessions(transactions_df, settings)
    transactions_df = enrich.enrich_device(transactions_df, users_df, settings)

    # Stage 3: Schema alignment
    events_df = align.align_to_schema(transactions_df)

    # Stage 4: LightFM dataset construction (includes temporal split)
    dataset_artifacts = build_dataset.build_lightfm_dataset(events_df, games_df, settings)

    with _mlflow_run_ctx(settings):
        _mlflow_log_params(settings, {
            "no_components": settings.NO_COMPONENTS,
            "loss": settings.LOSS,
            "learning_rate": settings.LEARNING_RATE,
            "item_alpha": settings.ITEM_ALPHA,
            "user_alpha": settings.USER_ALPHA,
            "n_epochs": settings.N_EPOCHS,
            "n_users": len(dataset_artifacts["active_users"]),
            "n_items": len(dataset_artifacts["active_items"]),
            "train_nnz": dataset_artifacts["train_interactions"].nnz,
            "test_nnz": dataset_artifacts["test_interactions"].nnz,
        })

        # Stage 5: Training — returns {"model": ..., "history": [...]}
        train_result = train.train_model(dataset_artifacts, settings)
        model = train_result["model"]
        training_history = train_result["history"]

        # Stage 6: Evaluation
        metrics = evaluate.evaluate_model(model, dataset_artifacts, settings)
        _mlflow_log_metrics(settings, metrics)

        # Stage 6b: Diagnostic plots (best-effort, never blocks training/export)
        if settings.MLFLOW_ENABLED and settings.EDA_ENABLED:
            try:
                with tempfile.TemporaryDirectory() as plot_dir:
                    generate_diagnostic_plots(
                        dataset_artifacts=dataset_artifacts,
                        events_df=events_df,
                        training_history=training_history,
                        output_dir=plot_dir,
                        max_games_to_plot=settings.EDA_MAX_GAMES_TO_PLOT,
                    )
                    _mlflow_log_diagnostics(settings, plot_dir)
            except Exception as exc:
                logger.warning("Diagnostic plot generation failed (non-fatal): %s", exc)

        # Stage 7: Export artifacts
        artifact_path = export.export_artifacts(model, dataset_artifacts, games_df, settings)
        _mlflow_log_artifacts(settings, artifact_path)

        # Stage 8: Log registry-compatible model and register it (if enabled)
        if settings.MLFLOW_ENABLED and settings.MLFLOW_REGISTRY_ENABLED:
            model_uri = _mlflow_log_registry_model(settings, artifact_path)
            if model_uri is not None:
                _register_model(
                    settings,
                    model_uri,
                    metrics,
                    dataset_artifacts["train_interactions"].nnz,
                )

    # Stage 9: Write features to Redis (if enabled, best-effort)
    export.write_features_to_redis(dataset_artifacts, settings)

    logger.info("Training flow completed successfully; artifacts exported to %s", artifact_path)

    return {
        "metrics": metrics,
        "artifact_path": artifact_path,
    }
