from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rag.config import COLLECTION_NAME, DEFAULT_TOP_K, SEMANTIC_CANDIDATES
from rag.ingestion import embed_content_with_fallback, get_collection


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    semantic_score: float
    keyword_score: float
    combined_score: float


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def embed_query(question: str) -> list[float]:
    return embed_content_with_fallback(question, task_type="retrieval_query")


def cosine_distance_to_score(distance: float | int | None) -> float:
    if distance is None:
        return 0.0
    # Chroma returns smaller distances for better matches. Convert to a bounded similarity-like score.
    return max(0.0, min(1.0, 1.0 - float(distance)))


def keyword_overlap_score(question: str, chunk_text: str) -> float:
    query_terms = tokenize(question)
    if not query_terms:
        return 0.0
    chunk_terms = tokenize(chunk_text)
    overlap = query_terms.intersection(chunk_terms)
    return len(overlap) / math.sqrt(len(query_terms) * max(len(chunk_terms), 1))


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    collection = get_collection(collection_name)
    if collection.count() == 0:
        return []

    query_embedding = embed_query(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, SEMANTIC_CANDIDATES),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        semantic_score = cosine_distance_to_score(distance)
        keyword_score = keyword_overlap_score(question, text)
        combined_score = (0.8 * semantic_score) + (0.2 * min(keyword_score * 3, 1.0))
        chunks.append(
            RetrievedChunk(
                text=text,
                metadata=metadata,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
            )
        )

    return sorted(chunks, key=lambda chunk: chunk.combined_score, reverse=True)[:top_k]


def estimate_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0

    best = chunks[0].combined_score
    supporting_sources = len(
        {
            (chunk.metadata.get("document"), chunk.metadata.get("page"))
            for chunk in chunks
            if chunk.combined_score >= best * 0.75
        }
    )
    support_bonus = min(0.12, supporting_sources * 0.03)
    return round(max(0.0, min(0.98, best + support_bonus)), 2)
