"""
Document loading and parsing module.
Handles extraction of text from PDF, DOCX, and TXT files,
and splits text into overlapping chunks for embedding.
"""

from typing import List

from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import SUPPORTED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────
# Text Extractors
# ─────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """Extract all text from a DOCX file."""
    doc = DocxDocument(file_path)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_txt(file_path: str) -> str:
    """Read a plain-text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()



# ─────────────────────────────────────────────────────────────────────
# Extractor Registry
# ─────────────────────────────────────────────────────────────────────
EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
}


def load_document(file_path: str, extension: str) -> str:
    """Dispatch to the correct extractor based on file extension. Falls back to plain text reading for other types."""
    extractor = EXTRACTORS.get(extension.lower())
    if extractor is None:
        return extract_text_from_txt(file_path)
    return extractor(file_path)


# ─────────────────────────────────────────────────────────────────────
# Text Chunking
# ─────────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
