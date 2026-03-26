from datetime import datetime, timezone
from itertools import islice

from prefect import task

from pipeline.config import PipelineSettings
from pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def _get_neo4j_driver(settings: PipelineSettings):
    """Create Neo4j driver. Import here to keep neo4j optional."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def _ensure_constraints(driver, database: str, player_key: str, game_key: str) -> None:
    """Create uniqueness constraints if they don't exist."""
    constraints = [
        f"CREATE CONSTRAINT player_id_unique IF NOT EXISTS FOR (p:Player) REQUIRE p.{player_key} IS UNIQUE",
        f"CREATE CONSTRAINT game_id_unique IF NOT EXISTS FOR (g:Game) REQUIRE g.{game_key} IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for stmt in constraints:
            session.run(stmt)
    logger.info("Neo4j constraints ensured")


def _iter_batches(rows: list, batch_size: int):
    iterator = iter(rows)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def _extract_embeddings(model, dataset_artifacts: dict) -> dict:
    """Extract player and game embeddings + biases from the trained LightFM model.

    Uses feature matrices to get full hybrid representations, matching what
    model.predict() uses internally during inference.
    """
    user_features_matrix = dataset_artifacts["user_features_matrix"]
    item_features_matrix = dataset_artifacts["item_features_matrix"]
    active_users = dataset_artifacts["active_users"]
    active_items = dataset_artifacts["active_items"]

    user_biases, user_embeddings = model.get_user_representations(
        features=user_features_matrix
    )
    item_biases, item_embeddings = model.get_item_representations(
        features=item_features_matrix
    )

    return {
        "user_embeddings": user_embeddings,
        "user_biases": user_biases,
        "item_embeddings": item_embeddings,
        "item_biases": item_biases,
        "active_users": list(active_users),
        "active_items": list(active_items),
    }


def _prepare_player_rows(emb_data: dict, player_key: str) -> list:
    rows = []
    for i, player_id in enumerate(emb_data["active_users"]):
        rows.append({
            player_key: str(player_id),
            "embedding": emb_data["user_embeddings"][i].tolist(),
            "bias": float(emb_data["user_biases"][i]),
        })
    return rows


def _prepare_game_rows(emb_data: dict, game_key: str) -> list:
    rows = []
    for i, game_id in enumerate(emb_data["active_items"]):
        rows.append({
            game_key: str(game_id),
            "embedding": emb_data["item_embeddings"][i].tolist(),
            "bias": float(emb_data["item_biases"][i]),
        })
    return rows


def _write_player_embeddings(
    driver, database: str, player_rows: list, player_key: str
) -> int:
    query = f"""
    UNWIND $rows AS row
    MATCH (p:Player {{{player_key}: row.{player_key}}})
    SET p.embedding = row.embedding,
        p.bias = row.bias,
        p.updated_at = datetime($updated_at)
    RETURN count(p) AS matched
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            rows=player_rows,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        record = result.single()
        return int(record["matched"]) if record else 0


def _write_game_embeddings(driver, database: str, game_rows: list, game_key: str) -> int:
    query = f"""
    UNWIND $rows AS row
    MATCH (g:Game {{{game_key}: row.{game_key}}})
    SET g.embedding = row.embedding,
        g.bias = row.bias,
        g.updated_at = datetime($updated_at)
    RETURN count(g) AS matched
    """
    with driver.session(database=database) as session:
        result = session.run(
            query,
            rows=game_rows,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        record = result.single()
        return int(record["matched"]) if record else 0


@task
def export_embeddings_to_neo4j(
    model,
    dataset_artifacts: dict,
    settings: PipelineSettings,
    player_key: str = "id",
    game_key: str = "id",
) -> int:
    """Extract LightFM embeddings and push them to existing Neo4j nodes.

    Best-effort: logs warning and returns 0 if Neo4j is unavailable or disabled.
    Returns total Player + Game nodes updated.

    player_key and game_key must match the primary key properties used on
    Player and Game nodes in the target graph.
    """
    if not settings.NEO4J_ENABLED:
        print("Neo4j disabled, skipping embedding export")
        return 0

    driver = None
    try:
        driver = _get_neo4j_driver(settings)
        driver.verify_connectivity()
        database = settings.NEO4J_DATABASE

        _ensure_constraints(driver, database, player_key, game_key)

        emb_data = _extract_embeddings(model, dataset_artifacts)
        player_rows = _prepare_player_rows(emb_data, player_key)
        game_rows = _prepare_game_rows(emb_data, game_key)

        players_written = 0
        for batch in _iter_batches(player_rows, settings.NEO4J_BATCH_SIZE):
            players_written += _write_player_embeddings(driver, database, batch, player_key)

        games_written = 0
        for batch in _iter_batches(game_rows, settings.NEO4J_BATCH_SIZE):
            games_written += _write_game_embeddings(driver, database, batch, game_key)

        total_updated = players_written + games_written
        print(
            f"Neo4j embedding export: {players_written} players, "
            f"{games_written} games updated"
        )
        logger.info(
            "Neo4j embedding export complete: players=%s games=%s",
            players_written,
            games_written,
        )
        return total_updated

    except Exception as exc:
        logger.warning("Neo4j embedding export failed (best-effort): %s", exc)
        print(f"WARNING: Neo4j export failed (best-effort): {exc}")
        return 0
    finally:
        if driver is not None:
            driver.close()
