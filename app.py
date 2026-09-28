from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentencemodel import (
    load_pdf,
    load_csv,
    semantic_chunk,
    upload_to_qdrant,
    search_qdrant,
    search_qdrant_with_tags,
    clear_qdrant_collection,
    generate_answer,
    generate_answer_with_context,
)

app = FastAPI(
    title="Financial RAG API",
    version="2.0.0"
)


# -----------------------------
# Request Models
# -----------------------------

class UploadRequest(BaseModel):
    file_path: str
    source: str | None = None


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


def _source_name(file_path: str) -> str:
    return Path(file_path).name.lower()


def _serialize_results(results):
    return [
        {
            "unique_id": r.id,
            "source": (r.payload or {}).get("source"),
            "tags": (r.payload or {}).get("tags"),
            "score": r.score,
        }
        for r in results
    ]


# -----------------------------
# Upload
# -----------------------------

@app.post("/aethal/upload/pdf")
def upload_pdf(request: UploadRequest):
    try:

        texts = load_pdf(request.file_path)

        nodes = semantic_chunk(texts)

        unique_ids = upload_to_qdrant(
            nodes,
            source=request.source or _source_name(request.file_path),
            file_path=request.file_path,
        )

        return {
            "status": "success",
            "message": "PDF uploaded successfully.",
            "source": request.source or _source_name(request.file_path),
            "chunks_uploaded": len(unique_ids),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/aethal/upload/csv")
def upload_csv(request: UploadRequest):
    try:

        rows = load_csv(request.file_path)

        # Same pipeline as PDF: semantic-chunk the rows, then upload.
        nodes = semantic_chunk(rows)

        unique_ids = upload_to_qdrant(
            nodes,
            source=request.source or _source_name(request.file_path),
            file_path=request.file_path,
        )

        return {
            "status": "success",
            "message": "CSV uploaded successfully.",
            "source": request.source or _source_name(request.file_path),
            "chunks_uploaded": len(unique_ids),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/aethal/reset-collection")
def reset_collection():
    try:
        result = clear_qdrant_collection()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# RAG Question Answering
# -----------------------------

@app.post("/ask-aethal-finance")
def ask_rag(request: QuestionRequest):
    try:

        results = search_qdrant(
            request.question,
            top_k=request.top_k
        )

        answer = generate_answer_with_context(
            request.question,
            results
        )

        return {
            "status": "success",
            "search_type": "vector",
            "question": request.question,
            "answer": answer,
            "sources": _serialize_results(results),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/ask-aethal-finance/tags")
def ask_rag_with_tags(request: QuestionRequest):
    try:

        # search_qdrant_with_tags derives its own tags from the question
        # via KeyBERT, tag-filters first, then fills any remaining slots
        # with plain vector search.
        results = search_qdrant_with_tags(
            request.question,
            top_k=request.top_k,
        )

        answer = generate_answer_with_context(
            request.question,
            results
        )

        return {
            "status": "success",
            "search_type": "tag_filtered",
            "question": request.question,
            "answer": answer,
            "sources": _serialize_results(results),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------
# Groq Only (no retrieval)
# -----------------------------

@app.post("/ask-groq")
def ask_groq(request: QuestionRequest):
    try:

        answer = generate_answer(request.question)

        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )