from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
SESSIONS_DIR = PROJECT_ROOT / "data" / "chat_sessions"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "enterprise_knowledge")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "EMBEDDING_FALLBACK_MODELS",
        "gemini-embedding-001,text-embedding-004,text-embedding-005",
    ).split(",")
    if model.strip()
]
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")
GENERATION_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GENERATION_FALLBACK_MODELS",
        "gemini-2.5-flash,gemini-2.5-flash-preview-05-20,gemini-2.0-flash,gemini-2.0-flash-exp",
    ).split(",")
    if model.strip()
]

CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "500"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
SEMANTIC_CANDIDATES = int(os.getenv("SEMANTIC_CANDIDATES", "20"))

