from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag.config import SESSIONS_DIR
from rag.generator import answer_question
from rag.ingestion import get_collection, ingest_documents, reset_index


st.set_page_config(page_title="RAGbot", layout="wide")

st.title("RAGbot")
st.caption("upload and ask away!")

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_FILE = SESSIONS_DIR / "sessions.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_sessions() -> list[dict]:
    if not SESSIONS_FILE.exists():
        return []
    return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))


def save_sessions(sessions: list[dict]) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def messages_path(session_id: str) -> Path:
    return SESSIONS_DIR / session_id / "messages.json"


def load_session_messages(session_id: str) -> list[dict]:
    path = messages_path(session_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_session_messages(session_id: str, messages: list[dict]) -> None:
    path = messages_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")


def create_session(title: str = "New chat") -> dict:
    session_id = uuid.uuid4().hex[:12]
    session_dir = SESSIONS_DIR / session_id
    documents_dir = session_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "id": session_id,
        "title": title,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "collection_name": f"kb_{session_id}",
        "documents_dir": str(documents_dir),
    }
    sessions = load_sessions()
    sessions.insert(0, session)
    save_sessions(sessions)
    save_session_messages(session_id, [])
    return session


def get_session(session_id: str) -> dict:
    for session in load_sessions():
        if session["id"] == session_id:
            return session
    return create_session()


def update_session(session_id: str, **updates) -> None:
    sessions = load_sessions()
    for session in sessions:
        if session["id"] == session_id:
            session.update(updates)
            session["updated_at"] = now_iso()
            break
    save_sessions(sessions)


def ensure_active_session() -> dict:
    sessions = load_sessions()
    if not sessions:
        session = create_session()
    else:
        session = sessions[0]

    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = session["id"]
        st.session_state.messages = load_session_messages(session["id"])

    return get_session(st.session_state.active_session_id)


def clear_chat() -> None:
    st.session_state.messages = []
    save_session_messages(st.session_state.active_session_id, [])


def clear_saved_documents(documents_dir: Path) -> int:
    documents_dir.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for path in documents_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            path.unlink()
            deleted += 1
    return deleted


def clear_knowledge_base(session: dict) -> None:
    clear_saved_documents(Path(session["documents_dir"]))
    reset_index(session["collection_name"])
    clear_chat()
    st.session_state.uploader_key += 1


def save_uploaded_files(uploaded_files, documents_dir: Path) -> list[Path]:
    documents_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        destination = documents_dir / uploaded_file.name
        with destination.open("wb") as file:
            shutil.copyfileobj(uploaded_file, file)
        saved_paths.append(destination)
    return saved_paths


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

active_session = ensure_active_session()

with st.sidebar:
    st.header("Chats")
    sessions = load_sessions()
    session_ids = [session["id"] for session in sessions]
    session_labels = {
        session["id"]: f"{session['title']} ({session['updated_at'][:10]})" for session in sessions
    }
    current_index = session_ids.index(st.session_state.active_session_id)
    selected_session_id = st.selectbox(
        "Open chat",
        session_ids,
        index=current_index,
        format_func=lambda session_id: session_labels[session_id],
    )

    if selected_session_id != st.session_state.active_session_id:
        save_session_messages(st.session_state.active_session_id, st.session_state.messages)
        st.session_state.active_session_id = selected_session_id
        st.session_state.messages = load_session_messages(selected_session_id)
        st.session_state.uploader_key += 1
        st.rerun()

    if st.button("New chat"):
        save_session_messages(st.session_state.active_session_id, st.session_state.messages)
        new_session = create_session()
        st.session_state.active_session_id = new_session["id"]
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

    active_session = get_session(st.session_state.active_session_id)
    active_documents_dir = Path(active_session["documents_dir"])

    st.divider()
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload internal PDFs, markdown, or text documents.",
        key=f"document_uploader_{st.session_state.uploader_key}",
    )

    replace_existing = st.toggle(
        "Replace this chat's knowledge base on ingest",
        value=True,
        help="When enabled, only the files currently selected above will be indexed for this chat.",
    )

    if st.button("Ingest / Rebuild Index", type="primary"):
        with st.spinner("Extracting text, chunking, embedding, and indexing documents..."):
            try:
                if uploaded_files and replace_existing:
                    clear_saved_documents(active_documents_dir)
                if uploaded_files:
                    saved = save_uploaded_files(uploaded_files, active_documents_dir)
                    st.info(f"Saved {len(saved)} file(s) to this chat.")
                    title = ", ".join(path.name for path in saved[:2])
                    if len(saved) > 2:
                        title += f" + {len(saved) - 2} more"
                    update_session(active_session["id"], title=title)

                result = ingest_documents(
                    documents_dir=active_documents_dir,
                    reset=True,
                    collection_name=active_session["collection_name"],
                )
                st.success(result["message"])
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    saved_documents = sorted(
        path.name
        for path in active_documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if saved_documents:
        with st.expander("This chat's document files"):
            for document_name in saved_documents:
                st.write(f"- {document_name}")

    try:
        collection = get_collection(active_session["collection_name"])
        st.metric("Indexed chunks", collection.count())
    except Exception:
        st.metric("Indexed chunks", 0)

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=5)
    use_memory = st.toggle("Conversation memory", value=True)

    if st.button("Clear current chat messages"):
        clear_chat()
        st.rerun()

    if st.button("Clear this chat's knowledge base"):
        clear_knowledge_base(active_session)
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(
                        f"- {source['document']} | page {source['page']} | "
                        f"chunk {source.get('chunk', 'n/a')} | score {source['score']}"
                    )
            st.caption(f"Confidence: {message.get('confidence', 0):.2f}")


question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    save_session_messages(st.session_state.active_session_id, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and generating answer..."):
            try:
                history = st.session_state.messages if use_memory else []
                result = answer_question(
                    question,
                    top_k=top_k,
                    chat_history=history,
                    collection_name=active_session["collection_name"],
                )
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("Sources", expanded=True):
                        for source in result["sources"]:
                            st.write(
                                f"- {source['document']} | page {source['page']} | "
                                f"chunk {source.get('chunk', 'n/a')} | score {source['score']}"
                            )
                st.caption(f"Confidence: {result['confidence']:.2f}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "confidence": result["confidence"],
                    }
                )
                save_session_messages(st.session_state.active_session_id, st.session_state.messages)
            except Exception as exc:
                error = f"Answer generation failed: {exc}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
                save_session_messages(st.session_state.active_session_id, st.session_state.messages)
