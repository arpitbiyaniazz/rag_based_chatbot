"""
Retrieval Evaluator — Context Relevance Verification with Retry & Ranking
=========================================================================
After documents are retrieved from the vector store, an LLM judges
whether the chunks are **capable of answering** the user's question.

Retry strategy (up to 3 attempts):
  Attempt 1: original top_k
  Attempt 2: top_k × 2  (cast a wider net)
  Attempt 3: top_k × 3 + MMR search (maximum diversity)

Once relevant chunks are found, a **ranking step** selects only the
top 2 most relevant chunks to pass as context — reducing noise.

If all 3 attempts fail, the pipeline returns "I don't know".

Public API:
    evaluate_and_retry_retrieval(question, vector_store, ...) → RetrievalResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS

from llm import query_openai, query_huggingface, query_mistral


# ─────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """Outcome of the evaluated retrieval process."""

    context_chunks: List[str]
    sources: List[dict]
    is_relevant: bool
    attempts_made: int
    chunks_before_ranking: int = 0
    chunks_after_ranking: int = 0
    attempt_log: List[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# LLM relevance judgement
# ─────────────────────────────────────────────────────────────────────
_RELEVANCE_SYSTEM_PROMPT = """\
You are a **Retrieval Quality Judge** for a RAG system.

You will receive:
- A user QUESTION
- A set of RETRIEVED DOCUMENT CHUNKS

Your ONLY job is to decide whether the retrieved chunks contain \
enough information to answer the question meaningfully.

## Rules
- The chunks do NOT need to contain the *full* answer — partial \
  coverage is acceptable as long as they provide useful context.
- If the chunks are completely off-topic, or contain no information \
  relevant to the question, answer NO.
- Be lenient: if at least one chunk has reasonable relevance, answer YES.

## Response format — STRICT
Return EXACTLY these two lines (no markdown, no extra text):
RELEVANT: YES or NO
REASON: <one-line explanation>
"""


def _judge_relevance(
    question: str,
    chunks: List[str],
    llm_provider: str,
    llm_model: str,
    api_key: str,
) -> Tuple[bool, str]:
    """
    Ask the LLM whether the retrieved chunks can answer the question.

    Returns:
        (is_relevant, reason_string)
    """
    if not chunks:
        return (False, "No chunks retrieved.")

    # Build a concise summary of the chunks for the judge
    chunk_display = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c[:600]}" for i, c in enumerate(chunks)
    )

    user_prompt = (
        f"### Question\n{question}\n\n"
        f"### Retrieved Document Chunks\n{chunk_display}"
    )

    try:
        if llm_provider == "OpenAI":
            raw = query_openai(
                user_prompt, llm_model, api_key,
                system_prompt=_RELEVANCE_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=150,
            )
        elif llm_provider == "Mistral AI":
            raw = query_mistral(
                user_prompt, llm_model, api_key,
                system_prompt=_RELEVANCE_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=150,
            )
        else:
            raw = query_huggingface(
                user_prompt, llm_model, api_key,
                system_prompt=_RELEVANCE_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=150,
            )
    except Exception:
        # If the LLM judge fails, assume relevant to avoid blocking
        return (True, "Relevance check unavailable — assuming relevant.")

    return _parse_relevance_response(raw)


def _parse_relevance_response(raw: str) -> Tuple[bool, str]:
    """Parse the structured LLM relevance judgement."""
    is_relevant = True  # default to permissive
    reason = ""

    for line in raw.strip().splitlines():
        line = line.strip()
        upper = line.upper()

        if upper.startswith("RELEVANT:"):
            value = line.split(":", 1)[1].strip().upper()
            is_relevant = value in ("YES", "TRUE", "Y")

        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return (is_relevant, reason)


# ─────────────────────────────────────────────────────────────────────
# Layer 3 — LLM-based chunk ranking (pick top N)
# ─────────────────────────────────────────────────────────────────────
_RANKING_SYSTEM_PROMPT = """\
You are a **Chunk Relevance Ranker** for a RAG system.

You will receive:
- A user QUESTION
- A numbered list of DOCUMENT CHUNKS

Your ONLY job is to identify which chunks are MOST relevant to \
answering the question and return their numbers ranked from most \
relevant to least relevant.

## Rules
- Return ONLY the chunk numbers, comma-separated, ordered from \
  most relevant to least relevant.
- Include ALL chunks in your ranking even if some are less relevant.
- Do NOT add any explanation or extra text.

## Response format — STRICT
Return EXACTLY one line:
RANKING: 3, 1, 4, 2  (example — your actual numbers will differ)
"""

DEFAULT_TOP_N = 2


def _rank_chunks(
    question: str,
    chunks: List[str],
    sources: List[dict],
    llm_provider: str,
    llm_model: str,
    api_key: str,
    top_n: int = DEFAULT_TOP_N,
) -> Tuple[List[str], List[dict]]:
    """
    Ask the LLM to rank retrieved chunks by relevance and return
    only the top_n most relevant ones.

    Returns:
        (filtered_chunks, filtered_sources)
    """
    if len(chunks) <= top_n:
        return (chunks, sources)

    chunk_display = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c[:600]}" for i, c in enumerate(chunks)
    )

    user_prompt = (
        f"### Question\n{question}\n\n"
        f"### Document Chunks (total: {len(chunks)})\n{chunk_display}"
    )

    try:
        if llm_provider == "OpenAI":
            raw = query_openai(
                user_prompt, llm_model, api_key,
                system_prompt=_RANKING_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=100,
            )
        elif llm_provider == "Mistral AI":
            raw = query_mistral(
                user_prompt, llm_model, api_key,
                system_prompt=_RANKING_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=100,
            )
        else:
            raw = query_huggingface(
                user_prompt, llm_model, api_key,
                system_prompt=_RANKING_SYSTEM_PROMPT,
                temperature=0.0, max_tokens=100,
            )
    except Exception:
        # If ranking fails, return the first top_n chunks as fallback
        return (chunks[:top_n], sources[:top_n])

    ranked_indices = _parse_ranking_response(raw, len(chunks))

    # Take only the top_n indices
    selected = ranked_indices[:top_n]

    filtered_chunks = [chunks[i] for i in selected]
    filtered_sources = [sources[i] for i in selected]

    return (filtered_chunks, filtered_sources)


def _parse_ranking_response(raw: str, total_chunks: int) -> List[int]:
    """
    Parse the LLM ranking response into a list of 0-based indices.

    Falls back to sequential order [0, 1, 2, ...] on parse failure.
    """
    import re

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("RANKING:"):
            nums_str = line.split(":", 1)[1].strip()
            # Extract all integers from the string
            nums = re.findall(r"\d+", nums_str)
            indices = []
            for n in nums:
                idx = int(n) - 1  # convert 1-based to 0-based
                if 0 <= idx < total_chunks and idx not in indices:
                    indices.append(idx)
            if indices:
                return indices

    # Fallback: return original order
    return list(range(total_chunks))


# ─────────────────────────────────────────────────────────────────────
# Retrieval strategies per attempt
# ─────────────────────────────────────────────────────────────────────
def _retrieve_attempt(
    question: str,
    vector_store: FAISS,
    base_top_k: int,
    attempt: int,
) -> Tuple[List[str], List[dict]]:
    """
    Execute a retrieval attempt with escalating parameters.

    Attempt 0: standard similarity search with base top_k
    Attempt 1: doubled top_k for wider coverage
    Attempt 2: tripled top_k + MMR for maximum diversity
    """
    if attempt == 0:
        k = base_top_k
        results = vector_store.similarity_search_with_score(question, k=k)

    elif attempt == 1:
        k = base_top_k * 2
        results = vector_store.similarity_search_with_score(question, k=k)

    else:  # attempt == 2
        k = base_top_k * 3
        # Use MMR (Maximal Marginal Relevance) for diverse results
        try:
            mmr_docs = vector_store.max_marginal_relevance_search(
                question, k=k, fetch_k=k * 2,
            )
            # MMR doesn't return scores, so we assign a placeholder
            results = [(doc, 0.0) for doc in mmr_docs]
        except Exception:
            # Fall back to similarity search if MMR fails
            results = vector_store.similarity_search_with_score(question, k=k)

    chunks = [doc.page_content for doc, _score in results]
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "score": round(float(score), 4),
            "type": "document",
        }
        for doc, score in results
    ]

    return (chunks, sources)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
MAX_RETRIEVAL_ATTEMPTS = 3


def evaluate_and_retry_retrieval(
    question: str,
    vector_store: FAISS,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    top_k: int = 4,
) -> RetrievalResult:
    """
    Retrieve documents, judge their relevance, and retry with
    escalating parameters if the context is insufficient.

    Parameters
    ----------
    question : str
        The (sanitized) user question.
    vector_store : FAISS
        The active FAISS vector store.
    llm_provider, llm_model, api_key : str
        LLM configuration for the relevance judge.
    top_k : int
        Base number of chunks to retrieve.

    Returns
    -------
    RetrievalResult
        Contains the best context found (or empty if all attempts fail).
    """
    attempt_log: List[dict] = []

    for attempt in range(MAX_RETRIEVAL_ATTEMPTS):
        chunks, sources = _retrieve_attempt(
            question, vector_store, top_k, attempt,
        )

        is_relevant, reason = _judge_relevance(
            question, chunks, llm_provider, llm_model, api_key,
        )

        strategy_names = [
            f"similarity (top_k={top_k})",
            f"similarity (top_k={top_k * 2})",
            f"MMR (top_k={top_k * 3})",
        ]

        attempt_log.append({
            "attempt": attempt + 1,
            "strategy": strategy_names[attempt],
            "chunks_retrieved": len(chunks),
            "is_relevant": is_relevant,
            "reason": reason,
        })

        if is_relevant:
            # Rank and select only the top 2 most relevant chunks
            chunks_before = len(chunks)
            ranked_chunks, ranked_sources = _rank_chunks(
                question, chunks, sources,
                llm_provider, llm_model, api_key,
                top_n=DEFAULT_TOP_N,
            )

            return RetrievalResult(
                context_chunks=ranked_chunks,
                sources=ranked_sources,
                is_relevant=True,
                attempts_made=attempt + 1,
                chunks_before_ranking=chunks_before,
                chunks_after_ranking=len(ranked_chunks),
                attempt_log=attempt_log,
            )

    # All 3 attempts failed — return empty context
    return RetrievalResult(
        context_chunks=[],
        sources=[],
        is_relevant=False,
        attempts_made=MAX_RETRIEVAL_ATTEMPTS,
        attempt_log=attempt_log,
    )
