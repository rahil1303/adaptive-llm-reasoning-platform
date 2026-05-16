"""
Critique evaluation routes.
"""

from fastapi import APIRouter
from app.models.schemas import CritiqueRequest, CritiqueResult
from app.services.critique import critique_response

router = APIRouter()


@router.post("/evaluate", response_model=CritiqueResult)
async def evaluate(req: CritiqueRequest):
    """Run critique evaluation on a response."""
    result = await critique_response(
        query=req.query,
        response_text=req.response_text,
        context_chunks=req.context_chunks,
        model_id=req.model_id,
    )
    return CritiqueResult(**result)
