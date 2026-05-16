"""
Model management routes.
"""

from fastapi import APIRouter
from app.config import AVAILABLE_MODELS

router = APIRouter()


@router.get("/list")
async def list_models():
    """List all available models with their configuration."""
    models = []
    for key, config in AVAILABLE_MODELS.items():
        models.append({
            "id": key,
            "provider": config.provider,
            "display_name": config.display_name,
            "model_id": config.model_id,
            "has_api_key": config.api_key is not None,
        })
    return {"models": models}
