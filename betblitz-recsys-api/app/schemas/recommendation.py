from pydantic import BaseModel, Field
from typing import List


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    exclude_played: bool = True


class GameRecommendation(BaseModel):
    game_id: str
    game_name: str
    game_type: str
    provider: str
    score: float
    rank: int


class RecommendMetadata(BaseModel):
    model_version: str
    top_k: int
    excluded_played: bool
    is_cold_start: bool
    source: str


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[GameRecommendation]
    metadata: RecommendMetadata
