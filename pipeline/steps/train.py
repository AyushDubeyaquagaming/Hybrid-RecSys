from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from prefect import task

from pipeline.config import PipelineSettings
from pipeline.exceptions import ModelTrainingError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


@task
def train_model(dataset_artifacts: dict, settings: PipelineSettings) -> LightFM:
    logger.info(
        "Starting model training with train_nnz=%s test_nnz=%s epochs=%s",
        dataset_artifacts["train_interactions"].nnz,
        dataset_artifacts["test_interactions"].nnz,
        settings.N_EPOCHS,
    )
    model = LightFM(
        no_components=settings.NO_COMPONENTS,
        loss=settings.LOSS,
        learning_rate=settings.LEARNING_RATE,
        item_alpha=settings.ITEM_ALPHA,
        user_alpha=settings.USER_ALPHA,
        random_state=settings.SEED,
    )

    train_interactions = dataset_artifacts["train_interactions"]
    train_weights = dataset_artifacts["train_weights"]
    test_interactions = dataset_artifacts["test_interactions"]
    user_features_matrix = dataset_artifacts["user_features_matrix"]
    item_features_matrix = dataset_artifacts["item_features_matrix"]

    try:
        for epoch in range(settings.N_EPOCHS):
            model.fit_partial(
                interactions=train_interactions,
                sample_weight=train_weights,
                user_features=user_features_matrix,
                item_features=item_features_matrix,
                num_threads=settings.PREDICT_NUM_THREADS,
                epochs=1,
            )

            if (epoch + 1) % 5 == 0:
                tr_p = precision_at_k(
                    model,
                    train_interactions,
                    user_features=user_features_matrix,
                    item_features=item_features_matrix,
                    k=settings.EVAL_K,
                    num_threads=settings.PREDICT_NUM_THREADS,
                ).mean()
                te_p = precision_at_k(
                    model,
                    test_interactions,
                    train_interactions=train_interactions,
                    user_features=user_features_matrix,
                    item_features=item_features_matrix,
                    k=settings.EVAL_K,
                    num_threads=settings.PREDICT_NUM_THREADS,
                ).mean()
                print(
                    f"Epoch {epoch+1:2d}/{settings.N_EPOCHS} | "
                    f"Train P@{settings.EVAL_K}: {tr_p:.4f} | "
                    f"Test P@{settings.EVAL_K}: {te_p:.4f}"
                )
                if settings.MLFLOW_ENABLED:
                    import mlflow
                    mlflow.log_metrics(
                        {"train_p5": float(tr_p), "test_p5": float(te_p)},
                        step=epoch + 1,
                    )
    except Exception as exc:
        raise ModelTrainingError(f"LightFM training failed: {exc}") from exc

    print("Training complete ✅")
    return model
