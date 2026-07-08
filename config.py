"""
Configuration constants for the RAG Document Chatbot.
Centralizes all model lists, supported formats, and search modes.
"""

# ─────────────────────────────────────────────────────────────────────
# Supported Document Types
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# ─────────────────────────────────────────────────────────────────────
# Embedding Models
# ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2 (Fast, Good Quality)": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (Best Quality)": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-MiniLM-L3-v2 (Fastest)": "sentence-transformers/paraphrase-MiniLM-L3-v2",
}

# ─────────────────────────────────────────────────────────────────────
# LLM Providers & Models
# ─────────────────────────────────────────────────────────────────────
LLM_PROVIDERS = ["OpenAI", "HuggingFace Inference",]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-3.5-turbo",
]

HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
]


# ─────────────────────────────────────────────────────────────────────
# Search Modes
# ─────────────────────────────────────────────────────────────────────
SEARCH_MODES = [
    "📄 Documents Only",
    "🌐 Web Search Only",
    "📄 + 🌐 Documents & Web",
]
