from pymongo import MongoClient

from pipeline.config import PipelineSettings
from pipeline.exceptions import ExternalServiceError


def get_db(settings: PipelineSettings):
    try:
        client = MongoClient(
            settings.MONGO_URI,
            directConnection=settings.MONGO_DIRECT_CONNECTION,
            serverSelectionTimeoutMS=settings.MONGO_TIMEOUT_MS,
        )
        client.admin.command("ping")
        return client[settings.MONGO_DB]
    except Exception as exc:
        raise ExternalServiceError(
            f"Failed to connect to MongoDB at {settings.MONGO_URI}: {exc}"
        ) from exc
