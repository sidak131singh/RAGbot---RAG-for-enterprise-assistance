from __future__ import annotations

import json
import re
from typing import Any

from google.genai import types

from rag.config import COLLECTION_NAME, GENERATION_FALLBACK_MODELS, GENERATION_MODEL
from rag.ingestion import get_gemini_client
from rag.retriever import RetrievedChunk, estimate_confidence, retrieve


SYSTEM_INSTRUCTIONS = """You are an Enterprise Knowledge Assistant.
Answer only from the provided context.
If the context does not contain the answer, say: "I could not find this information in the indexed documents."
If the user asks a broad question such as "what does this document say", "summarize this", or "what is this about", summarize the retrieved context instead of saying the information is unavailable.
Be concise, practical, and avoid guessing.
Do not invent policies, numbers, dates, names, or source references.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        document = chunk.metadata.get("document", "Unknown document")
        page = chunk.metadata.get("page", "Unknown page")
        context_blocks.append(
            f"[Source {index}: {document}, page {page}, score {chunk.combined_score:.2f}]\n{chunk.text}"
        )
    return "\n\n".join(context_blocks)


def dedupe_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    seen: set[tuple[str, Any]] = set()
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        document = chunk.metadata.get("document", "Unknown document")
        page = chunk.metadata.get("page", "Unknown page")
        key = (document, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "document": document,
                "page": page,
                "chunk": chunk.metadata.get("chunk"),
                "score": round(chunk.combined_score, 2),
            }
        )
    return sources


def build_prompt(question: str, chunks: list[RetrievedChunk], chat_history: list[dict] | None = None) -> str:
    history_text = ""
    if chat_history:
        recent_turns = chat_history[-6:]
        history_text = "\n".join(
            f"{turn.get('role', 'user').title()}: {turn.get('content', '')}" for turn in recent_turns
        )

    return f"""{SYSTEM_INSTRUCTIONS}

Conversation history:
{history_text or "No prior conversation."}

Retrieved context:
{format_context(chunks) or "No context retrieved."}

Employee question:
{question}

Instructions:
- For specific factual questions, answer the exact question using the context.
- For broad document-summary questions, provide a short summary of the retrieved document content.
- If the retrieved context is unrelated to the question, set answer_found to false.

Return JSON only with this schema:
{{
  "answer": "concise grounded answer",
  "answer_found": true
}}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return {"answer": text.strip(), "answer_found": True}


def generation_model_candidates() -> list[str]:
    candidates = [GENERATION_MODEL, *GENERATION_FALLBACK_MODELS]
    return list(dict.fromkeys(candidates))


def generate_with_fallback(prompt: str) -> str:
    client = get_gemini_client()
    errors: list[str] = []
    for model in generation_model_candidates():
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=700,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError("No configured Gemini generation model worked. Tried: " + " | ".join(errors))


def answer_question(
    question: str,
    top_k: int = 5,
    chat_history: list[dict] | None = None,
    min_confidence: float = 0.25,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    chunks = retrieve(question, top_k=top_k, collection_name=collection_name)
    confidence = estimate_confidence(chunks)
    sources = dedupe_sources(chunks)

    if not chunks or confidence < min_confidence:
        return {
            "answer": "I could not find this information in the indexed documents.",
            "sources": sources,
            "confidence": confidence,
            "answer_found": False,
        }

    parsed = parse_json_response(generate_with_fallback(build_prompt(question, chunks, chat_history)))
    answer_found = bool(parsed.get("answer_found", True))
    answer = parsed.get("answer") or "I could not find this information in the indexed documents."

    if not answer_found:
        answer = "I could not find this information in the indexed documents."
        sources = []
        confidence = min(confidence, 0.2)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "answer_found": answer_found,
    }
