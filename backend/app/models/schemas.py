"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Chat ---

class ChatRequest(BaseModel):
    query: str
    model_ids: list[str] = Field(default=["llama3-70b"], description="Models to query")
    mode: str = Field(default="direct", description="Interaction mode")
    use_rag: bool = Field(default=True, description="Whether to use RAG retrieval")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    similarity_metric: str = Field(default="cosine", description="cosine | l2 | dot")


class ChunkResult(BaseModel):
    text: str
    score: float
    source: str
    chunk_index: int


class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    response: str
    tokens_used: int
    latency_ms: float
    chunks_used: list[ChunkResult] = []
    mode: str = "direct"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatResponse(BaseModel):
    query: str
    responses: list[ModelResponse]
    session_id: str


# --- Documents ---

class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    total_tokens: int
    ingested_at: str


# --- Critique ---

class CritiqueRequest(BaseModel):
    query: str
    response_text: str
    context_chunks: list[str] = []
    model_id: str = "llama-3.3-70b"


class CritiqueResult(BaseModel):
    correctness_score: float = Field(ge=0, le=1)
    groundedness_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    issues: list[str] = []
    suggestions: list[str] = []
    summary: str = ""


# --- Insights ---

class InsightRequest(BaseModel):
    query: str
    responses: list[dict]
    model_id: str = "llama-3.3-70b"


class InsightResult(BaseModel):
    summary: str
    key_topics: list[str]
    reasoning_quality: str
    model_comparison: str = ""
    metadata: dict = {}
