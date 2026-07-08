"""
LLM integration module.
Provides wrappers for OpenAI and HuggingFace Inference API calls.
"""

from openai import OpenAI
from huggingface_hub import InferenceClient


# ─────────────────────────────────────────────────────────────────────
# Default System Prompt
# ─────────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based only "
    "on the provided context. If the answer is not found in the "
    "context, say so clearly. Be concise and accurate."
)


# ─────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────
def query_openai(
    prompt: str,
    model: str,
    api_key: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Call the OpenAI Chat Completions API."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────
# HuggingFace Inference
# ─────────────────────────────────────────────────────────────────────
def query_huggingface(
    prompt: str,
    model: str,
    api_key: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Call the HuggingFace Inference API."""
    client = InferenceClient(token=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
