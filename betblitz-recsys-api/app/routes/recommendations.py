from fastapi import APIRouter, HTTPException, Request

from app.metrics import COLD_START_TOTAL, ITEMS_RETURNED, RECOMMEND_LATENCY, RECOMMEND_REQUESTS
from app.schemas.recommendation import (
    GameRecommendation,
    RecommendMetadata,
    RecommendRequest,
    RecommendResponse,
)
from app.services.model_service import MODEL_VERSION

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: Request, body: RecommendRequest):
    model_service = request.app.state.model_service

    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    with RECOMMEND_LATENCY.time():
        result = model_service.recommend(
            user_id=body.user_id,
            top_k=body.top_k,
            exclude_played=body.exclude_played,
        )

    RECOMMEND_REQUESTS.labels(source=result["source"]).inc()
    ITEMS_RETURNED.observe(len(result["recommendations"]))
    if result["is_cold_start"]:
        COLD_START_TOTAL.inc()

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
