from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.services.model_service import MODEL_VERSION

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    model_service = request.app.state.model_service

    if not model_service.is_loaded():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False},
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "n_users": model_service.n_users,
        "n_items": model_service.n_items,
    }
