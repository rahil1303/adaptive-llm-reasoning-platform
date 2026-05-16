"""
Insights generation routes.
"""

from fastapi import APIRouter
from app.models.schemas import InsightRequest, InsightResult
from app.services.insights import generate_insights

router = APIRouter()


@router.post("/generate", response_model=InsightResult)
async def generate(req: InsightRequest):
    """Generate insights from model responses."""
    result = await generate_insights(
        query=req.query,
        responses=req.responses,
        model_id=req.model_id,
    )
    return InsightResult(**result)
