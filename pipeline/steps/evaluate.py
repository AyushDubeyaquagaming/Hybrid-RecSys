import numpy as np
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
from prefect import task

from pipeline.config import PipelineSettings
from pipeline.exceptions import DataValidationError
from pipeline.logging_utils import get_logger


logger = get_logger(__name__)


@task
def evaluate_model(
    model: LightFM, dataset_artifacts: dict, settings: PipelineSettings
) -> dict:
    train_interactions = dataset_artifacts["train_interactions"]
    test_interactions = dataset_artifacts["test_interactions"]
    user_features_matrix = dataset_artifacts["user_features_matrix"]
    item_features_matrix = dataset_artifacts["item_features_matrix"]

    k = settings.EVAL_K
    num_threads = settings.PREDICT_NUM_THREADS
    if train_interactions.nnz == 0:
        raise DataValidationError("Cannot evaluate model with an empty training interaction matrix.")
    logger.info(
        "Evaluating model with train_shape=%s test_shape=%s k=%s",
        train_interactions.shape,
        test_interactions.shape,
        k,
    )

    final_train_p5 = precision_at_k(
        model,
        train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=k,
        num_threads=num_threads,
    ).mean()

    final_test_p5 = precision_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=k,
        num_threads=num_threads,
    ).mean()

    final_train_auc = auc_score(
        model,
        train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        num_threads=num_threads,
    ).mean()

    final_test_auc = auc_score(
        model,
        test_interactions,
        train_interactions=train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        num_threads=num_threads,
    ).mean()

    # NDCG@5 on users with test interactions (excluding known train items)
    n_users_eval, n_items_eval = test_interactions.shape
    test_csr_eval = csr_matrix(test_interactions)
    train_csr_eval = csr_matrix(train_interactions)
    ndcg_vals = []
    for user_idx in range(n_users_eval):
        true_items = test_csr_eval.getrow(user_idx).indices
        if len(true_items) == 0:
            continue
        scores = model.predict(
            user_ids=np.repeat(user_idx, n_items_eval),
            item_ids=np.arange(n_items_eval),
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=num_threads,
        )

        known_train = train_csr_eval.getrow(user_idx).indices
        scores_ndcg = scores.copy()
        if len(known_train) > 0:
            scores_ndcg[known_train] = -1e9

        y_true = np.zeros(n_items_eval)
        y_true[true_items] = 1
        ndcg_vals.append(ndcg_score([y_true], [scores_ndcg], k=k))

    final_ndcg5 = float(np.mean(ndcg_vals)) if len(ndcg_vals) > 0 else float("nan")

    metrics = {
        "train_precision_at_k": float(final_train_p5),
        "test_precision_at_k": float(final_test_p5),
        "train_auc": float(final_train_auc),
        "test_auc": float(final_test_auc),
        f"ndcg_at_{k}": final_ndcg5,
    }

    print("=" * 52)
    print("LightFM Evaluation Summary")
    print("=" * 52)
    print(f"Train Precision@{k}: {final_train_p5:.4f}")
    print(f"Test  Precision@{k}: {final_test_p5:.4f}")
    print(f"Train AUC:         {final_train_auc:.4f}")
    print(f"Test  AUC:         {final_test_auc:.4f}")
    print(f"NDCG@{k}:            {final_ndcg5:.4f}")
    print(f"Generalization gap: {(final_train_p5 - final_test_p5):.4f}")
    logger.info("Evaluation metrics: %s", metrics)

    return metrics
