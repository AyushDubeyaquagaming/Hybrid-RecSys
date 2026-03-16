from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.services.model_service import MODEL_VERSION

router = APIRouter()
_settings = Settings()


def _check_redis() -> bool:
    """Ping Redis and return True if reachable, False otherwise."""
    try:
        import redis
        r = redis.Redis(
            host=_settings.REDIS_HOST,
            port=_settings.REDIS_PORT,
            db=_settings.REDIS_DB,
            password=_settings.REDIS_PASSWORD or None,
            socket_connect_timeout=1,
            decode_responses=True,
        )
        r.ping()
        return True
    except Exception:
        return False


@router.get("/health")
async def health(request: Request):
    model_service = request.app.state.model_service

    if not model_service.is_loaded():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False},
        )

    response: dict = {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "n_users": model_service.n_users,
        "n_items": model_service.n_items,
    }

    if _settings.REDIS_ENABLED:
        response["redis_connected"] = _check_redis()

    return response
