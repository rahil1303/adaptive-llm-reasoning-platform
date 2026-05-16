"""
Chat routes: handles queries, RAG retrieval, multi-model execution.
"""

import asyncio
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ModelResponse, ChunkResult
from app.services.llm_provider import call_llm, call_multiple
from app.services.vector_store import get_store
from app.services.ingestion import get_embedding
from app.services.logger import log_interaction, new_session_id
from app.config import INTERACTION_MODES

router = APIRouter()


@router.post("/query", response_model=ChatResponse)
async def query(req: ChatRequest):
    """
    Main query endpoint.
    1. Optionally retrieve context via RAG
    2. Build prompt with interaction mode
    3. Send to all requested models
    4. Return responses with metadata
    """
    session_id = new_session_id()

    # Log the query
    log_interaction(session_id, "query", {
        "query": req.query,
        "models": req.model_ids,
        "mode": req.mode,
        "use_rag": req.use_rag,
        "top_k": req.top_k,
        "similarity_metric": req.similarity_metric,
    })

    # RAG retrieval
    chunks_used = []
    context_text = ""
    if req.use_rag:
        try:
            store = get_store()
            if store.entries:
                query_emb = await get_embedding(req.query)
                results = store.search(
                    query_embedding=query_emb,
                    top_k=req.top_k,
                    metric=req.similarity_metric,
                )
                chunks_used = [
                    ChunkResult(
                        text=r["text"],
                        score=r["score"],
                        source=r["source"],
                        chunk_index=r["chunk_index"],
                    )
                    for r in results
                ]
                context_text = "\n\n---\n\n".join(r["text"] for r in results)
        except Exception as e:
            log_interaction(session_id, "error", {"stage": "retrieval", "error": str(e)})

    # Build messages with interaction mode
    mode_config = INTERACTION_MODES.get(req.mode, INTERACTION_MODES["direct"])
    system_prompt = mode_config["system_prompt"]

    if context_text:
        system_prompt += f"\n\nRelevant context from documents:\n{context_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.query},
    ]

    # Call all models
    raw_results = await call_multiple(req.model_ids, messages)

    responses = []
    for r in raw_results:
        resp = ModelResponse(
            model_id=r["model_id"],
            model_name=r["model_name"],
            response=r["text"],
            tokens_used=r["tokens_used"],
            latency_ms=r["latency_ms"],
            chunks_used=chunks_used,
            mode=req.mode,
        )
        responses.append(resp)

        # Log each response
        log_interaction(session_id, "response", {
            "model_id": r["model_id"],
            "tokens_used": r["tokens_used"],
            "latency_ms": r["latency_ms"],
            "finish_reason": r.get("finish_reason"),
        })

    return ChatResponse(
        query=req.query,
        responses=responses,
        session_id=session_id,
    )
