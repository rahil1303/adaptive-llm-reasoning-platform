# AI Reasoning Platform

Full-stack multi-model LLM interaction platform for studying reasoning support, retrieval behavior, and response quality across AI systems.

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and GROQ_API_KEY

# Run
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### 3. Using It

1. Open Settings → Upload your thesis PDF (or any document)
2. Select models to compare (need API keys set)
3. Choose an interaction mode (Direct / Hint-First / Guided Reasoning)
4. Configure retrieval settings (metric, Top-K)
5. Ask questions → see side-by-side responses
6. Hit "Critique" on any response to get automated evaluation

### Docker

```bash
# Copy env file
cp backend/.env.example backend/.env
# Edit with your keys

docker-compose up --build
```

## Architecture

```
backend/
  app/
    routes/        # API endpoints (chat, models, documents, critique, insights)
    services/      # Core logic (LLM provider, vector store, ingestion, critique, insights, logger)
    models/        # Pydantic schemas
    config.py      # Model configs, interaction modes
  data/            # Vectors, logs, uploads (persisted)

frontend/
  src/
    App.tsx        # Main UI — chat, comparison, retrieval inspector, critique panel
```

## API Keys

- **Groq** (free): https://console.groq.com — gives access to LLaMA 3, Qwen
- **OpenAI**: https://platform.openai.com — GPT models + embeddings

## Features

- Multi-model side-by-side comparison (LLaMA, Qwen, GPT)
- Configurable interaction modes (direct, hint-first, guided reasoning)
- RAG pipeline with semantic chunking and configurable similarity metrics
- Chunk inspection and retrieval visualization
- Multi-round critique engine with correctness/groundedness/completeness scoring
- Automated insight generation
- JSONL interaction logging for analysis
- Extensible — add new models by editing config.py
