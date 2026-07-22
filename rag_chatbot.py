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
        "current_file_config": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val




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
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
            st.rerun()

        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    return {
        "emb_model_id": emb_model_id,
        "chunk_size": 800,
        "chunk_overlap": 200,
        "top_k": 4,
        "web_results_count": 5,
        "temperature": 0.3,
        "max_tokens": 1024,
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


def handle_file_upload(emb_model_id: str, chunk_size: int, chunk_overlap: int):
    """
    Handle uploading a single document of any type.
    Whenever a new file is uploaded, a new chunking is created, replacing the previous one.
    """
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=None,  # supports any file type
        accept_multiple_files=False,
        help="Upload any file to chat with its contents.",
    )

    if not uploaded_file:
        # If no file is uploaded, clear the vector store and stats
        if st.session_state.vector_store is not None:
            st.session_state.vector_store = None
            st.session_state.processed_hashes = set()
            st.session_state.doc_store = {}
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
        return

    # Calculate hash of the uploaded file
    file_bytes = uploaded_file.getvalue()
    h = file_content_hash(file_bytes)

    # Check if this file has already been processed with the exact current embedding & chunk configuration
    current_config = (h, emb_model_id, chunk_size, chunk_overlap)
    if st.session_state.get("current_file_config") == current_config:
        return

    # A new file or a different embedding/chunking configuration has been loaded.
    # Completely reset and clear all old databases and session properties.
    st.session_state.vector_store = None
    st.session_state.processed_hashes = set()
    st.session_state.doc_store = {}
    st.session_state.doc_count = 0
    st.session_state.chunk_count = 0
    st.session_state.current_file_config = None

    with st.status("Processing uploaded document …", expanded=True) as status:
        name = uploaded_file.name
        ext = os.path.splitext(name)[1].lower()

        st.write(f"📄 Parsing **{name}** …")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            text = load_document(tmp_path, ext)
        except Exception as exc:
            st.error(f"Error reading **{name}**: {exc}")
            status.update(label="❌ Failed to parse document.", state="error")
            return
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not text.strip():
            st.warning(f"No extractable text in **{name}**.")
            status.update(label="⚠️ Empty document.", state="error")
            return

        st.write(f"✂️ Creating new chunking (chunk_size={chunk_size}) …")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        metas = [{"source": name, "chunk_index": i} for i in range(len(chunks))]

        st.write(f"🧠 Generating embeddings for {len(chunks)} chunks …")
        embeddings = get_embedding_model(emb_model_id)
        st.session_state.vector_store = build_vector_store(chunks, metas, embeddings)

        # Cache in doc_store and update session state
        st.session_state.doc_store[h] = {
            "name": name,
            "chunks": chunks,
            "metas": metas,
        }
        st.session_state.processed_hashes.add(h)
        st.session_state.doc_count = 1
        st.session_state.chunk_count = len(chunks)
        st.session_state.current_file_config = current_config

        status.update(label="✅ Document successfully chunked and indexed!", state="complete")



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
            if query_eval and query_eval.get("was_modified", False):
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
        page_title="ASK Document",
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
        "<h1>📚 ASK Document</h1>"
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
        "ASK Document · Powered by LangChain, FAISS, Sentence Transformers & DuckDuckGo · "
        "Documents are processed locally — only the LLM query is sent to the API."
    )


if __name__ == "__main__":
    main()
