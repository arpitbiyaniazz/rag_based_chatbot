"""
Embeddings and vector store module.
Handles loading Sentence-Transformer models and building/managing the FAISS index.
"""

from typing import List

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ─────────────────────────────────────────────────────────────────────
# Embedding Model (cached across Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model …")
def get_embedding_model(model_id: str) -> HuggingFaceEmbeddings:
    """Load and cache a Sentence-Transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────────────
# Vector Store Construction
# ─────────────────────────────────────────────────────────────────────
def build_vector_store(
    chunks: List[str],
    metadatas: List[dict],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Create a FAISS index from text chunks and their metadata."""
    return FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
    )
