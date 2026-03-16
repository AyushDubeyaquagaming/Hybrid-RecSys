from prefect import flow

from pipeline.config import PipelineSettings
from pipeline.logging_utils import get_logger
from pipeline.steps import align, build_dataset, enrich, evaluate, export, ingest, train


logger = get_logger(__name__)


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

    # Stage 5: Training
    model = train.train_model(dataset_artifacts, settings)

    # Stage 6: Evaluation
    metrics = evaluate.evaluate_model(model, dataset_artifacts, settings)

    # Stage 7: Export artifacts
    artifact_path = export.export_artifacts(model, dataset_artifacts, games_df, settings)

    logger.info("Training flow completed successfully; artifacts exported to %s", artifact_path)

    return {
        "metrics": metrics,
        "artifact_path": artifact_path,
    }
