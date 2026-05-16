"""
Document ingestion: reads documents, chunks semantically, embeds, and stores.
Uses local sentence-transformers for embeddings (free, no API key needed).
"""

import re
from pathlib import Path
from typing import Optional
from app.services.vector_store import get_store

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


async def get_embedding(text: str) -> list[float]:
    model = _get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embs]


def semantic_chunk(text: str, max_chunk_size: int = 800, overlap: int = 100) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > max_chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) + 1 <= max_chunk_size:
                        sub_chunk = (sub_chunk + " " + sent).strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return chunks


def read_document(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except ImportError:
            raise ValueError("PDF support requires pymupdf: pip install pymupdf")
    else:
        return path.read_text(encoding="utf-8", errors="replace")


async def ingest_document(file_path: str, max_chunk_size: int = 800) -> dict:
    text = read_document(file_path)
    chunks = semantic_chunk(text, max_chunk_size=max_chunk_size)

    source = Path(file_path).name

    store = get_store()
    total_tokens = 0
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = await get_embeddings_batch(batch)
        for j, (chunk_text, emb) in enumerate(zip(batch, embeddings)):
            store.add(
                text=chunk_text,
                embedding=emb,
                source=source,
                chunk_index=i + j,
            )
            total_tokens += len(chunk_text.split())

    return {
        "filename": source,
        "chunk_count": len(chunks),
        "total_tokens": total_tokens,
    }