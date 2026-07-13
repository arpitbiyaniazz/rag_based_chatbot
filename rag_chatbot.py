"""
RAG-Based Document Chatbot — Main Streamlit Application
========================================================
Entry point that assembles the UI from modular components.

Run with:  streamlit run rag_chatbot.py

Module structure:
  config.py          → Constants (models, providers, search modes)
  document_loader.py → PDF/DOCX/TXT parsing and text chunking
  vector_store.py    → Embedding model loading and FAISS index
  web_search.py      → DuckDuckGo web search
  llm.py             → OpenAI and HuggingFace LLM wrappers
  rag_pipeline.py    → RAG orchestration (retrieve → prompt → generate)
  styles.py          → Custom CSS for the Streamlit UI
"""

import os
import tempfile
import hashlib
from typing import List

import streamlit as st

# ── Internal modules ────────────────────────────────────────────────
from config import (
    SUPPORTED_EXTENSIONS,
    EMBEDDING_MODELS,
    LLM_PROVIDERS,
    OPENAI_MODELS,
    HF_MODELS,
    MISTRAL_MODELS,
    SEARCH_MODES,
)
from document_loader import load_document, chunk_text
from vector_store import get_embedding_model, build_vector_store
from rag_pipeline import ask_question
from styles import get_custom_css


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def file_content_hash(content: bytes) -> str:
    """Return a short SHA-256 hex digest for deduplication."""
    return hashlib.sha256(content).hexdigest()[:16]


def init_session_state():
    """Initialise Streamlit session-state keys."""
    defaults = {
        "vector_store": None,
        "processed_hashes": set(),
        "chat_history": [],
        "doc_count": 0,
        "chunk_count": 0,
        "api_key_input": "",
        "config_saved": False,
        # Per-document cache: {hash: {"name": str, "chunks": [...], "metas": [...]}}
        "doc_store": {},
        # Track the set of hashes from the last uploader state
        "last_upload_hashes": set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_source_pills(sources: list) -> str:
    """Build HTML source pills from a list of source dicts."""
    pills = []
    for s in sources:
        if s.get("type") == "web":
            url = s.get("url", "#")
            pills.append(
                f'<span class="source-pill-web">'
                f'🌐 <a href="{url}" target="_blank">{s["source"]}</a>'
                f'</span>'
            )
        else:
            score_str = f' (score: {s["score"]})' if "score" in s else ""
            pills.append(
                f'<span class="source-pill">📄 {s["source"]}{score_str}</span>'
            )
    return " ".join(pills)


# ─────────────────────────────────────────────────────────────────────
# UI Sections
# ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with advanced parameters and stats. Returns config dict."""
    with st.sidebar:
        st.markdown("### 🎛️ Advanced Parameters")

        emb_label = st.selectbox("Embedding Model", list(EMBEDDING_MODELS.keys()), index=0)
        emb_model_id = EMBEDDING_MODELS[emb_label]

        st.divider()

        chunk_size = st.slider("Chunk Size", 200, 2000, 800, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
        top_k = st.slider("Top-K Retrieved Chunks", 1, 10, 4)
        web_results_count = st.slider("Web Results Count", 1, 10, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.05)
        max_tokens = st.slider("Max Output Tokens", 128, 4096, 1024, step=128)

        st.divider()

        # Stats
        st.markdown("### 📊 Knowledge Base")
        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="stat-card"><div class="stat-value">{st.session_state.doc_count}</div>'
            f'<div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="stat-card"><div class="stat-value">{st.session_state.chunk_count}</div>'
            f'<div class="stat-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.processed_hashes = set()
            st.session_state.doc_store = {}
            st.session_state.last_upload_hashes = set()
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
            st.rerun()

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    return {
        "emb_model_id": emb_model_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "web_results_count": web_results_count,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def render_config_panel():
    """Render the main-area model & API configuration panel. Returns config dict."""
    with st.expander("⚙️ **Model & API Configuration**", expanded=not st.session_state.config_saved):
        cfg_col1, cfg_col2, cfg_col3 = st.columns([1.2, 1.2, 1])

        with cfg_col1:
            llm_provider = st.selectbox(
                "🏢 LLM Provider",
                LLM_PROVIDERS,
                index=0,
                help="Choose between OpenAI, HuggingFace Inference, or Mistral AI.",
            )
            if llm_provider == "OpenAI":
                model_list = OPENAI_MODELS
                key_placeholder = "sk-..."
                key_label = "🔑 OpenAI API Key"
            elif llm_provider == "Mistral AI":
                model_list = MISTRAL_MODELS
                key_placeholder = "..."
                key_label = "🔑 Mistral API Key"
            else:
                model_list = HF_MODELS
                key_placeholder = "hf_..."
                key_label = "🔑 HuggingFace API Token"

        with cfg_col2:
            llm_model = st.selectbox(
                "🤖 Model",
                model_list,
                index=0,
                help="Select the language model to use for generating answers.",
            )

        with cfg_col3:
            search_mode = st.selectbox(
                "🔍 Search Mode",
                SEARCH_MODES,
                index=0,
                help="Choose how the chatbot finds context for your questions.",
            )

        api_key = st.text_input(
            key_label,
            type="password",
            placeholder=key_placeholder,
            help="Your API key is never stored permanently. It's only kept in session memory.",
        )

        # Connection status + save
        scol1, scol2 = st.columns([3, 1])
        with scol1:
            if api_key:
                st.markdown(
                    f'<span class="status-connected">'
                    f'<span class="status-dot-green"></span>'
                    f'{llm_provider} — {llm_model}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-disconnected">'
                    '<span class="status-dot-red"></span>'
                    'No API key entered</span>',
                    unsafe_allow_html=True,
                )
        with scol2:
            if st.button("💾 Save Config", use_container_width=True):
                if api_key:
                    st.session_state.config_saved = True
                    st.toast("✅ Configuration saved!", icon="✅")
                    st.rerun()
                else:
                    st.toast("⚠️ Enter an API key first.", icon="⚠️")

    # Show active config summary when collapsed
    if st.session_state.config_saved and api_key:
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            st.markdown(
                f'<span class="status-connected">'
                f'<span class="status-dot-green"></span>'
                f'{llm_provider} → {llm_model}</span>',
                unsafe_allow_html=True,
            )
        with acol2:
            mode_class = "mode-docs"
            if "Web" in search_mode and "Doc" in search_mode:
                mode_class = "mode-both"
            elif "Web" in search_mode:
                mode_class = "mode-web"
            st.markdown(
                f'<span class="mode-badge {mode_class}">{search_mode}</span>',
                unsafe_allow_html=True,
            )

    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "search_mode": search_mode,
        "api_key": api_key,
    }


def _rebuild_vector_store_from_cache(emb_model_id: str):
    """Rebuild the FAISS vector store from all cached document chunks."""
    doc_store = st.session_state.doc_store

    if not doc_store:
        st.session_state.vector_store = None
        st.session_state.doc_count = 0
        st.session_state.chunk_count = 0
        return

    all_chunks: List[str] = []
    all_metas: List[dict] = []

    for doc_data in doc_store.values():
        all_chunks.extend(doc_data["chunks"])
        all_metas.extend(doc_data["metas"])

    if all_chunks:
        embeddings = get_embedding_model(emb_model_id)
        st.session_state.vector_store = build_vector_store(all_chunks, all_metas, embeddings)
    else:
        st.session_state.vector_store = None

    st.session_state.doc_count = len(doc_store)
    st.session_state.chunk_count = len(all_chunks)


def handle_file_upload(emb_model_id: str, chunk_size: int, chunk_overlap: int):
    """
    Handle document upload with incremental processing.

    - Only genuinely new files are chunked and embedded.
    - Removed files' chunks are dropped and the index is rebuilt from cache.
    - Previously processed files are never re-processed.
    """
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Drag & drop PDF, TXT, or DOCX files here.",
    )

    # Compute the current set of file hashes from the uploader
    current_hashes: dict[str, bytes] = {}  # hash → raw bytes
    current_names: dict[str, str] = {}     # hash → filename
    if uploaded_files:
        for f in uploaded_files:
            raw = f.getvalue()
            h = file_content_hash(raw)
            current_hashes[h] = raw
            current_names[h] = f.name

    current_hash_set = set(current_hashes.keys())
    cached_hash_set = set(st.session_state.doc_store.keys())

    # Detect additions and removals
    added_hashes = current_hash_set - cached_hash_set
    removed_hashes = cached_hash_set - current_hash_set

    # Nothing changed — skip
    if not added_hashes and not removed_hashes:
        return

    # ── Handle removals ─────────────────────────────────────────────
    if removed_hashes:
        removed_names = [
            st.session_state.doc_store[h]["name"]
            for h in removed_hashes
            if h in st.session_state.doc_store
        ]
        for h in removed_hashes:
            st.session_state.doc_store.pop(h, None)
            st.session_state.processed_hashes.discard(h)

        # Rebuild the vector store from remaining cached docs
        with st.status(f"Removing {len(removed_names)} document(s) …", expanded=True) as status:
            for name in removed_names:
                st.write(f"🗑️ Removing **{name}** …")
            _rebuild_vector_store_from_cache(emb_model_id)
            status.update(label=f"✅ Removed {len(removed_names)} document(s).", state="complete")

    # ── Handle additions ────────────────────────────────────────────
    if added_hashes:
        with st.status(f"Processing {len(added_hashes)} new document(s) …", expanded=True) as status:
            new_chunks_total: List[str] = []
            new_metas_total: List[dict] = []
            embeddings = get_embedding_model(emb_model_id)

            for h in added_hashes:
                name = current_names[h]
                raw_bytes = current_hashes[h]
                ext = os.path.splitext(name)[1].lower()

                st.write(f"📄 Parsing **{name}** …")

                if ext not in SUPPORTED_EXTENSIONS:
                    st.warning(f"Skipped unsupported file: {name}")
                    continue

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    text = load_document(tmp_path, ext)
                except Exception as exc:
                    st.error(f"Error reading **{name}**: {exc}")
                    continue
                finally:
                    if "tmp_path" in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                if not text.strip():
                    st.warning(f"No extractable text in **{name}** — skipping.")
                    continue

                st.write(f"✂️ Chunking **{name}** (chunk_size={chunk_size}) …")
                chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                metas = [{"source": name, "chunk_index": i} for i in range(len(chunks))]

                # Cache per-document chunks
                st.session_state.doc_store[h] = {
                    "name": name,
                    "chunks": chunks,
                    "metas": metas,
                }
                st.session_state.processed_hashes.add(h)

                new_chunks_total.extend(chunks)
                new_metas_total.extend(metas)

            if new_chunks_total:
                st.write(f"🧠 Embedding {len(new_chunks_total)} new chunks …")
                new_store = build_vector_store(new_chunks_total, new_metas_total, embeddings)

                if st.session_state.vector_store is not None:
                    st.session_state.vector_store.merge_from(new_store)
                else:
                    st.session_state.vector_store = new_store

                # Update counts from cache (always accurate)
                st.session_state.doc_count = len(st.session_state.doc_store)
                st.session_state.chunk_count = sum(
                    len(d["chunks"]) for d in st.session_state.doc_store.values()
                )
                status.update(label="✅ Documents processed!", state="complete")
            else:
                status.update(label="⚠️ No text extracted.", state="error")

    # Update the last-known upload hashes
    st.session_state.last_upload_hashes = current_hash_set


def render_chat_history():
    """Display all previous chat messages."""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                pills_html = render_source_pills(msg["sources"])
                st.markdown(f"**Sources:** {pills_html}", unsafe_allow_html=True)


def handle_question(config: dict, sidebar_config: dict):
    """Handle the user's chat input and run the RAG pipeline."""
    question = st.chat_input("Ask a question about your documents or the web …")

    if not question:
        return

    search_mode = config["search_mode"]
    api_key = config["api_key"]
    use_docs = "Documents" in search_mode or "📄" in search_mode
    use_web = "Web" in search_mode or "🌐" in search_mode

    # Guard: no knowledge base when docs mode is required
    if use_docs and not use_web and st.session_state.vector_store is None:
        st.warning(
            "⬆️ Please upload at least one document first, "
            "or switch to a search mode that includes Web Search."
        )
        st.stop()

    # Guard: no API key
    if not api_key:
        st.warning("🔑 Enter your API key in the configuration panel above before asking questions.")
        st.stop()

    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.markdown(
        f'<div class="chat-user">🧑 {question}</div>',
        unsafe_allow_html=True,
    )

    # RAG pipeline
    with st.spinner("Thinking …"):
        try:
            result = ask_question(
                question=question,
                vector_store=st.session_state.vector_store,
                llm_provider=config["llm_provider"],
                llm_model=config["llm_model"],
                api_key=api_key,
                search_mode=search_mode,
                top_k=sidebar_config["top_k"],
                web_results_count=sidebar_config["web_results_count"],
                temperature=sidebar_config["temperature"],
                max_tokens=sidebar_config["max_tokens"],
            )

            answer = result["answer"]
            sources = result["sources"]
            query_eval = result.get("query_evaluation", {})

            # ── Query evaluation feedback ───────────────────────────
            if query_eval:
                risk = query_eval.get("risk_level", "none")
                was_modified = query_eval.get("was_modified", False)

                if risk == "high" and not query_eval.get("is_safe", True):
                    st.error(
                        "🛡️ **Query blocked** — This query was identified as a "
                        "prompt-injection attempt and was not processed."
                    )
                elif risk in ("medium", "low") and query_eval.get("flags"):
                    flag_list = ", ".join(query_eval["flags"])
                    st.warning(
                        f"⚠️ **Security notice:** Potential issues detected "
                        f"(risk: {risk}). {query_eval.get('explanation', '')}"
                    )
                if was_modified:
                    st.info(
                        f"✨ **Query refined:** Your question was cleaned up "
                        f"for better results.\n\n"
                        f"**Original:** {query_eval.get('original_query', question)}\n\n"
                        f"**Used:** {query_eval.get('sanitized_query', question)}"
                    )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

            st.markdown(
                f'<div class="chat-assistant">🤖 {answer}</div>',
                unsafe_allow_html=True,
            )

            # Source pills
            if sources:
                pills_html = render_source_pills(sources)
                st.markdown(f"**Sources:** {pills_html}", unsafe_allow_html=True)

            # Show retrieved context in expanders
            if result["context"]:
                with st.expander("📖 Retrieved Document Context"):
                    for i, ctx in enumerate(result["context"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(ctx[:500] + (" …" if len(ctx) > 500 else ""))
                        st.divider()

            if result["web_results"]:
                with st.expander("🌐 Web Search Results"):
                    for r in result["web_results"]:
                        st.markdown(
                            f'<div class="web-result">'
                            f'<div class="web-title">{r["title"]}</div>'
                            f'<div class="web-url">{r["url"]}</div>'
                            f'<div class="web-snippet">{r["snippet"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Query security details expander
            if query_eval and query_eval.get("risk_level", "none") != "none":
                with st.expander("🛡️ Query Security Details"):
                    eval_col1, eval_col2 = st.columns(2)
                    with eval_col1:
                        risk_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(
                            query_eval.get("risk_level", "none"), "⚪"
                        )
                        st.markdown(f"**Risk Level:** {risk_emoji} {query_eval.get('risk_level', 'none').title()}")
                        st.markdown(f"**Safe:** {'✅ Yes' if query_eval.get('is_safe') else '❌ No'}")
                    with eval_col2:
                        st.markdown(f"**Modified:** {'Yes' if query_eval.get('was_modified') else 'No'}")
                        if query_eval.get("explanation"):
                            st.markdown(f"**Reason:** {query_eval['explanation']}")
                    if query_eval.get("flags"):
                        st.markdown("**Detected patterns:**")
                        for flag in query_eval["flags"]:
                            st.markdown(f"- `{flag}`")

            # Retrieval evaluation details
            retrieval_eval = result.get("retrieval_evaluation", {})
            if retrieval_eval:
                attempts = retrieval_eval.get("attempts_made", 0)
                is_relevant = retrieval_eval.get("is_relevant", True)
                attempt_log = retrieval_eval.get("attempt_log", [])
                before = retrieval_eval.get("chunks_before_ranking", 0)
                after = retrieval_eval.get("chunks_after_ranking", 0)

                if not is_relevant:
                    st.warning(
                        f"📊 **Retrieval failed** — Tried {attempts} retrieval "
                        f"strategies but could not find relevant documents."
                    )
                elif attempts > 1:
                    st.info(
                        f"🔄 **Retrieval refined** — Found relevant context on "
                        f"attempt {attempts} of 3."
                    )

                if before > 0 and after > 0 and before != after:
                    st.info(
                        f"🏆 **Chunk ranking** — Retrieved {before} chunks, "
                        f"selected the top {after} most relevant for context."
                    )

                if attempt_log:
                    with st.expander("📊 Retrieval Evaluation Details"):
                        for entry in attempt_log:
                            status = "✅" if entry.get("is_relevant") else "❌"
                            st.markdown(
                                f"**Attempt {entry.get('attempt', '?')}** — "
                                f"{entry.get('strategy', 'unknown')} → "
                                f"{status} {entry.get('chunks_retrieved', 0)} chunks"
                            )
                            if entry.get("reason"):
                                st.caption(f"  ↳ {entry['reason']}")
                        if before > 0 and after > 0:
                            st.divider()
                            st.markdown(
                                f"**Ranking:** {before} retrieved → "
                                f"**{after} selected** (top by relevance)"
                            )

        except Exception as exc:
            error_msg = f"Error generating answer: {exc}"
            st.error(error_msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"⚠️ {error_msg}", "sources": []}
            )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="RAG Document Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Inject custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Sidebar (advanced params)
    sidebar_config = render_sidebar()

    # Header
    st.markdown(
        '<div class="main-header">'
        "<h1>📚 RAG Document Chatbot</h1>"
        "<p>Upload documents · Ask questions · Search the web — all answers grounded in context.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Model & API config panel
    config = render_config_panel()

    st.markdown("")  # spacing

    # Document upload
    handle_file_upload(
        emb_model_id=sidebar_config["emb_model_id"],
        chunk_size=sidebar_config["chunk_size"],
        chunk_overlap=sidebar_config["chunk_overlap"],
    )

    # Chat history
    render_chat_history()

    # Question handling
    handle_question(config, sidebar_config)

    # Footer
    st.divider()
    st.caption(
        "RAG Document Chatbot · Powered by LangChain, FAISS, Sentence Transformers & DuckDuckGo · "
        "Documents are processed locally — only the LLM query is sent to the API."
    )


if __name__ == "__main__":
    main()
