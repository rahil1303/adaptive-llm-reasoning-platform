from dotenv import load_dotenv
load_dotenv()
"""
Adaptive AI Interaction & Reasoning Platform
Main FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat, models, documents, critique, insights

app = FastAPI(
    title="AI Reasoning Platform",
    description="Multi-model LLM interaction platform for studying reasoning and retrieval behavior",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(critique.router, prefix="/api/critique", tags=["critique"])
app.include_router(insights.router, prefix="/api/insights", tags=["insights"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
