### Tech Stack

**Backend**: Python, FastAPI, httpx, Pydantic, sentence-transformers, PyMuPDF, NumPy

**Frontend**: React, TypeScript, Vite

**Embeddings**: all-MiniLM-L6-v2 (local, no API calls — runs on CPU)

**LLM Providers**: Groq (LLaMA, Qwen — free tier), OpenAI (GPT models)

**Infrastructure**: Docker, Docker Compose, NGINX-ready

### Key Design Decisions

- **Local embeddings over API-based**: Using sentence-transformers locally eliminates embedding API costs and rate limits entirely. The all-MiniLM-L6-v2 model is ~90MB and runs fast on CPU.
- **JSONL vector store over ChromaDB/FAISS**: Fully inspectable, no binary dependencies, trivially portable.
- **OpenAI-compatible API format**: Both Groq and OpenAI use the same request format, so adding any compatible provider is a one-line config change.
- **Prompt-based interaction modes**: The three modes are implemented as system prompt templates, not separate code paths.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- API key from [Groq](https://console.groq.com) (free) and/or [OpenAI](https://platform.openai.com)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your actual API keys

uvicorn app.main:app --reload --port 8000
```

The first document upload will download the embedding model (~90MB, one-time).

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### 3. Using the Platform

1. Click **Settings** in the top right
2. Select one or more models (Groq models are free)
3. Click **Upload Document** and select a PDF, TXT, or MD file
4. Choose an interaction mode and retrieval settings
5. Type a question and hit **Send**
6. Click **Critique** on any response to get evaluation scores

### Docker

```bash
cp backend/.env.example backend/.env
# Fill in your API keys
docker-compose up --build
```

## What You Can Study With This

**Model behavior differences**: Same question, same context — how does LLaMA 3.3 70B compare to Qwen 3 32B? Where do they diverge? Which hallucinates more on technical content?

**Interaction mode effects**: Does hint-first prompting produce more grounded answers than direct response? Does guided reasoning improve completeness scores?

**Retrieval sensitivity**: How does changing the similarity metric affect which chunks get retrieved? Does Top-K=3 vs Top-K=10 change answer quality?

**Critique consistency**: Does the critique engine score the same response consistently across runs?

## Project Structure
