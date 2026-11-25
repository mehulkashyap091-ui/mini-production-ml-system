from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, description="Unique customer identifier")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of ranked items to return (1-50)")
    exclude_items: Optional[List[str]] = Field(
        default=None,
        description="Optional list of StockCodes to exclude (e.g., items already in cart)",
    )

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("user_id cannot be blank or whitespace-only")
        return cleaned

    @field_validator("exclude_items")
    @classmethod
    def normalize_exclude_items(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        return [item.strip().upper() for item in v if item and item.strip()]

class RecommendedItem(BaseModel):
    rank: int
    stock_code: str
    description: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: str
    top_k: int
    strategy: str
    is_cold_start: bool
    fallback_used: bool
    latency_ms: float
    model_version: str
    recommendations: List[RecommendedItem]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    catalog_items: int
    known_users: int
