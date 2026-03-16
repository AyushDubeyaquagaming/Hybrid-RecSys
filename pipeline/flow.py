from contextlib import nullcontext

from prefect import flow

from pipeline.config import PipelineSettings
from pipeline.logging_utils import get_logger
from pipeline.steps import align, build_dataset, enrich, evaluate, export, ingest, train


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

        # Stage 5: Training
        model = train.train_model(dataset_artifacts, settings)

        # Stage 6: Evaluation
        metrics = evaluate.evaluate_model(model, dataset_artifacts, settings)

        _mlflow_log_metrics(settings, metrics)

        # Stage 7: Export artifacts
        artifact_path = export.export_artifacts(model, dataset_artifacts, games_df, settings)

        _mlflow_log_artifacts(settings, artifact_path)

    # Stage 8: Write features to Redis (if enabled, best-effort)
    export.write_features_to_redis(dataset_artifacts, settings)

    logger.info("Training flow completed successfully; artifacts exported to %s", artifact_path)

    return {
        "metrics": metrics,
        "artifact_path": artifact_path,
    }
