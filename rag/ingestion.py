from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import fitz
from google import genai
from google.genai import types

from rag.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBEDDING_FALLBACK_MODELS,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict


def configure_gemini() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
        )


def get_gemini_client() -> genai.Client:
    configure_gemini()
    return genai.Client(api_key=GEMINI_API_KEY)


def get_collection(collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_index(collection_name: str = COLLECTION_NAME) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass
    client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_index, text))
    return pages


def extract_text_file(path: Path) -> list[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="ignore").strip())]


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> Iterable[tuple[Path, int, str]]:
    supported = {".pdf", ".txt", ".md"}
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue

        if path.suffix.lower() == ".pdf":
            pages = extract_pdf_pages(path)
        else:
            pages = extract_text_file(path)

        for page_number, text in pages:
            if text:
                yield path, page_number, text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_chunk_id(source_path: Path, page_number: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{source_path.name}:{page_number}:{chunk_index}:{text[:200]}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{source_path.stem}-{page_number}-{chunk_index}-{digest}"


def build_chunks(documents_dir: Path = DOCUMENTS_DIR) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source_path, page_number, text in load_documents(documents_dir):
        for chunk_index, chunk in enumerate(chunk_text(text), start=1):
            chunks.append(
                DocumentChunk(
                    id=build_chunk_id(source_path, page_number, chunk_index, chunk),
                    text=chunk,
                    metadata={
                        "document": source_path.name,
                        "source_path": str(source_path),
                        "page": page_number,
                        "chunk": chunk_index,
                    },
                )
            )
    return chunks


def embedding_model_candidates() -> list[str]:
    candidates = [EMBEDDING_MODEL, *EMBEDDING_FALLBACK_MODELS]
    return list(dict.fromkeys(candidates))


def embed_content_with_fallback(text: str, task_type: str) -> list[float]:
    client = get_gemini_client()
    errors: list[str] = []
    for model in embedding_model_candidates():
        try:
            response = client.models.embed_content(
                model=model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type.upper()),
            )
            return response.embeddings[0].values
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError(
        "No configured Gemini embedding model worked. Tried: " + " | ".join(errors)
    )


def embed_texts(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    embeddings: list[list[float]] = []
    for text in texts:
        embeddings.append(embed_content_with_fallback(text, task_type=task_type))
        time.sleep(0.05)
    return embeddings


def ingest_documents(
    documents_dir: Path = DOCUMENTS_DIR,
    reset: bool = True,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    documents_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(documents_dir)
    if not chunks:
        return {"documents": 0, "chunks": 0, "message": "No supported documents found."}

    if reset:
        reset_index(collection_name)

    collection = get_collection(collection_name)
    embeddings = embed_texts([chunk.text for chunk in chunks], task_type="retrieval_document")

    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )

    document_count = len({chunk.metadata["document"] for chunk in chunks})
    return {
        "documents": document_count,
        "chunks": len(chunks),
        "message": f"Indexed {len(chunks)} chunks from {document_count} document(s).",
    }


if __name__ == "__main__":
    result = ingest_documents()
    print(result["message"])
