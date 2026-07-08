"""
RAG orchestration module.
Ties together document retrieval, web search, prompt building, and LLM generation.
"""

from typing import List, Optional

from langchain_community.vectorstores import FAISS

from web_search import web_search_duckduckgo, format_web_results_as_context
from llm import query_openai, query_huggingface


# ─────────────────────────────────────────────────────────────────────
# Prompt Building
# ─────────────────────────────────────────────────────────────────────
def build_rag_prompt(
    question: str,
    context_chunks: List[str],
    web_chunks: Optional[List[str]] = None,
) -> str:
    """Create the RAG prompt with document context, optional web context, and question."""
    sections = []

    if context_chunks:
        doc_context = "\n\n---\n\n".join(context_chunks)
        sections.append(f"### Document Context\n{doc_context}")

    if web_chunks:
        web_context = "\n\n---\n\n".join(web_chunks)
        sections.append(f"### Web Search Results\n{web_context}")

    combined = "\n\n".join(sections) if sections else "(No context available.)"

    return (
        "Use the following context to answer the question. "
        "Cite your sources (document name or URL) when possible. "
        "If the answer cannot be determined from the context, state that clearly.\n\n"
        f"{combined}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer"
    )


# ─────────────────────────────────────────────────────────────────────
# Full RAG Pipeline
# ─────────────────────────────────────────────────────────────────────
def ask_question(
    question: str,
    vector_store: Optional[FAISS],
    llm_provider: str,
    llm_model: str,
    api_key: str,
    search_mode: str,
    top_k: int = 4,
    web_results_count: int = 5,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict:
    """Full RAG pipeline: retrieve → (web search) → prompt → generate."""

    context_chunks = []
    sources = []
    web_results = []

    use_docs = "Documents" in search_mode or "📄" in search_mode
    use_web = "Web" in search_mode or "🌐" in search_mode

    # 1. Document retrieval
    if use_docs and vector_store is not None:
        results = vector_store.similarity_search_with_score(question, k=top_k)
        context_chunks = [doc.page_content for doc, _score in results]
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "score": round(float(score), 4),
                "type": "document",
            }
            for doc, score in results
        ]

    # 2. Web search
    web_chunks = []
    if use_web:
        web_results = web_search_duckduckgo(question, max_results=web_results_count)
        web_chunks = format_web_results_as_context(web_results)
        for r in web_results:
            sources.append({
                "source": r["title"][:40] + ("…" if len(r["title"]) > 40 else ""),
                "url": r["url"],
                "type": "web",
            })

    # 3. Build prompt
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "The context may include document excerpts and/or web search results. "
        "Cite your sources when possible. If the answer cannot be determined, say so clearly. "
        "Be concise and accurate."
    )

    prompt = build_rag_prompt(question, context_chunks, web_chunks if use_web else None)

    # 4. Generate answer
    if llm_provider == "OpenAI":
        answer = query_openai(prompt, llm_model, api_key, system_prompt, temperature, max_tokens)
    else:
        answer = query_huggingface(prompt, llm_model, api_key, system_prompt, temperature, max_tokens)

    return {
        "answer": answer,
        "sources": sources,
        "context": context_chunks,
        "web_results": web_results,
    }
