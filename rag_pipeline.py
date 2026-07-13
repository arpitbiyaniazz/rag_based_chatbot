"""
RAG orchestration module.
Ties together document retrieval, web search, prompt building, and LLM generation.
"""

from typing import List, Optional

from langchain_community.vectorstores import FAISS

from web_search import web_search_duckduckgo, format_web_results_as_context
from llm import query_openai, query_huggingface, query_mistral
from query_evaluator import evaluate_query
from retrieval_evaluator import evaluate_and_retry_retrieval


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
    """Full RAG pipeline: evaluate → retrieve (with retry) → (web search) → prompt → generate."""

    context_chunks = []
    sources = []
    web_results = []
    retrieval_eval_info = {}

    # 0. Query evaluation — prompt-injection detection & refinement
    query_eval = evaluate_query(
        query=question,
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
    )

    # Block high-risk injection attempts outright
    if not query_eval.is_safe and query_eval.risk_level == "high":
        return {
            "answer": (
                "⚠️ **Your query was flagged as a potential prompt-injection attempt "
                "and has been blocked.**\n\n"
                f"**Reason:** {query_eval.explanation}\n\n"
                "Please rephrase your question to focus on your document or topic."
            ),
            "sources": [],
            "context": [],
            "web_results": [],
            "query_evaluation": {
                "original_query": query_eval.original_query,
                "risk_level": query_eval.risk_level,
                "flags": query_eval.flags,
                "is_safe": query_eval.is_safe,
                "was_modified": query_eval.was_modified,
                "explanation": query_eval.explanation,
            },
            "retrieval_evaluation": {},
        }

    # Use the sanitized / refined query for downstream processing
    safe_question = query_eval.sanitized_query

    use_docs = "Documents" in search_mode or "📄" in search_mode
    use_web = "Web" in search_mode or "🌐" in search_mode

    # 1. Document retrieval with relevance evaluation & retry
    if use_docs and vector_store is not None:
        retrieval_result = evaluate_and_retry_retrieval(
            question=safe_question,
            vector_store=vector_store,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key,
            top_k=top_k,
        )

        retrieval_eval_info = {
            "is_relevant": retrieval_result.is_relevant,
            "attempts_made": retrieval_result.attempts_made,
            "attempt_log": retrieval_result.attempt_log,
            "chunks_before_ranking": retrieval_result.chunks_before_ranking,
            "chunks_after_ranking": retrieval_result.chunks_after_ranking,
        }

        if retrieval_result.is_relevant:
            context_chunks = retrieval_result.context_chunks
            sources = retrieval_result.sources
        else:
            # All 3 retrieval attempts failed — documents can't answer
            # If web search is also not enabled, return "I don't know"
            if not use_web:
                return {
                    "answer": (
                        "🤷 **I don't know.**\n\n"
                        "I retrieved documents from your knowledge base "
                        f"across **{retrieval_result.attempts_made} attempts** "
                        "with different search strategies, but none of the "
                        "retrieved content was relevant enough to answer "
                        "your question.\n\n"
                        "**Suggestions:**\n"
                        "- Try rephrasing your question\n"
                        "- Upload additional documents that cover this topic\n"
                        "- Switch to a search mode that includes Web Search"
                    ),
                    "sources": [],
                    "context": [],
                    "web_results": [],
                    "query_evaluation": {
                        "original_query": query_eval.original_query,
                        "sanitized_query": query_eval.sanitized_query,
                        "risk_level": query_eval.risk_level,
                        "flags": query_eval.flags,
                        "is_safe": query_eval.is_safe,
                        "was_modified": query_eval.was_modified,
                        "explanation": query_eval.explanation,
                    },
                    "retrieval_evaluation": retrieval_eval_info,
                }
            # If web search IS enabled, continue without doc context
            # (web results may still answer the question)

    # 2. Web search
    web_chunks = []
    if use_web:
        web_results = web_search_duckduckgo(safe_question, max_results=web_results_count)
        web_chunks = format_web_results_as_context(web_results)
        for r in web_results:
            sources.append({
                "source": r["title"][:40] + ("…" if len(r["title"]) > 40 else ""),
                "url": r["url"],
                "type": "web",
            })

    # If no context at all (no docs and no web results), return "I don't know"
    if not context_chunks and not web_chunks:
        return {
            "answer": (
                "🤷 **I don't know.**\n\n"
                "No relevant context was found from either documents or web search "
                "to answer your question. Please try rephrasing or uploading "
                "relevant documents."
            ),
            "sources": [],
            "context": [],
            "web_results": [],
            "query_evaluation": {
                "original_query": query_eval.original_query,
                "sanitized_query": query_eval.sanitized_query,
                "risk_level": query_eval.risk_level,
                "flags": query_eval.flags,
                "is_safe": query_eval.is_safe,
                "was_modified": query_eval.was_modified,
                "explanation": query_eval.explanation,
            },
            "retrieval_evaluation": retrieval_eval_info,
        }

    # 3. Build prompt
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "The context may include document excerpts and/or web search results. "
        "Cite your sources when possible. If the answer cannot be determined, say so clearly. "
        "Be concise and accurate."
    )

    prompt = build_rag_prompt(safe_question, context_chunks, web_chunks if use_web else None)

    # 4. Generate answer
    if llm_provider == "OpenAI":
        answer = query_openai(prompt, llm_model, api_key, system_prompt, temperature, max_tokens)
    elif llm_provider == "Mistral AI":
        answer = query_mistral(prompt, llm_model, api_key, system_prompt, temperature, max_tokens)
    else:
        answer = query_huggingface(prompt, llm_model, api_key, system_prompt, temperature, max_tokens)

    return {
        "answer": answer,
        "sources": sources,
        "context": context_chunks,
        "web_results": web_results,
        "query_evaluation": {
            "original_query": query_eval.original_query,
            "sanitized_query": query_eval.sanitized_query,
            "risk_level": query_eval.risk_level,
            "flags": query_eval.flags,
            "is_safe": query_eval.is_safe,
            "was_modified": query_eval.was_modified,
            "explanation": query_eval.explanation,
        },
        "retrieval_evaluation": retrieval_eval_info,
    }

