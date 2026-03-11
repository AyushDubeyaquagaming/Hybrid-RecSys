from fastapi import APIRouter, Request, HTTPException
from app.schemas.recommendation import (
    RecommendRequest,
    RecommendResponse,
    GameRecommendation,
    RecommendMetadata,
)
from app.services.model_service import MODEL_VERSION

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: Request, body: RecommendRequest):
    model_service = request.app.state.model_service

    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = model_service.recommend(
        user_id=body.user_id,
        top_k=body.top_k,
        exclude_played=body.exclude_played,
    )

    return RecommendResponse(
        user_id=body.user_id,
        recommendations=[GameRecommendation(**r) for r in result["recommendations"]],
        metadata=RecommendMetadata(
            model_version=MODEL_VERSION,
            top_k=body.top_k,
            excluded_played=body.exclude_played,
            is_cold_start=result["is_cold_start"],
            source=result["source"],
        ),
    )
