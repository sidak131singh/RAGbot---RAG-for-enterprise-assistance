from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.generator import answer_question
from rag.ingestion import ingest_documents


app = FastAPI(title="Enterprise Knowledge Assistant API")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    document: str
    page: int | str
    chunk: int | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    answer_found: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest() -> dict[str, Any]:
    try:
        return ingest_documents(reset=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, Any]:
    try:
        return answer_question(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

