# 📚 RAG Document Chatbot

A **Retrieval-Augmented Generation** chatbot built with Streamlit. Upload your documents (PDF, TXT, DOCX), and ask natural-language questions — answers are grounded in the content you provide. Optionally augment with live web search.

---

## Features

| Feature | Details |
|---|---|
| **Multi-format upload** | PDF, TXT, DOCX — drag & drop multiple files at once |
| **Local embeddings** | Sentence-Transformers models run on your machine (no embedding API cost) |
| **FAISS vector store** | Fast, in-memory similarity search |
| **Dual LLM backends** | OpenAI (GPT-4o, GPT-3.5-turbo, …) or HuggingFace Inference API |
| **Web search** | DuckDuckGo-powered search (no API key needed) — use alone or combined with docs |
| **3 search modes** | Documents Only · Web Only · Documents & Web |
| **Configurable pipeline** | Chunk size, overlap, top-K retrieval, temperature, max tokens |
| **Source citations** | Each answer shows which document chunks / web results were used |
| **Dark glassmorphism UI** | Modern, polished Streamlit theme |

---

## Quick Start

### 1. Clone & install

```bash
cd "RAg based chatbot"
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

### 2. Set your API key

You can enter the key directly in the main configuration panel, or export it:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# OR HuggingFace
export HF_TOKEN="hf_..."
```

### 3. Run

```bash
streamlit run rag_chatbot.py
```

The app opens at **http://localhost:8501**.

---

## Project Structure

```
RAg based chatbot/
├── rag_chatbot.py       # Main Streamlit entry point (UI assembly)
├── config.py            # Constants: models, providers, search modes
├── document_loader.py   # PDF/DOCX/TXT parsing & text chunking
├── vector_store.py      # Embedding model loading & FAISS index
├── web_search.py        # DuckDuckGo web search (no API key needed)
├── llm.py               # OpenAI & HuggingFace LLM wrappers
├── rag_pipeline.py      # RAG orchestration (retrieve → prompt → generate)
├── styles.py            # Custom CSS for the Streamlit UI
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | All constants — model lists, supported extensions, search modes |
| `document_loader.py` | Extract text from PDF/DOCX/TXT; split into overlapping chunks |
| `vector_store.py` | Load Sentence-Transformer embeddings; build & manage FAISS index |
| `web_search.py` | DuckDuckGo search with HTML scraping fallback; format results for LLM |
| `llm.py` | Thin wrappers around OpenAI and HuggingFace chat completion APIs |
| `rag_pipeline.py` | Orchestrate retrieval + web search + prompt building + LLM generation |
| `styles.py` | Return the full CSS `<style>` block for the glassmorphism dark theme |
| `rag_chatbot.py` | Streamlit UI — assembles sidebar, config panel, upload, chat, and footer |

---

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  Upload Docs │────▶│  Parse & Chunk│────▶│  Embed (ST)   │
└──────────────┘     └──────────────┘     └──────┬────────┘
                                                  │
                                           ┌──────▼────────┐
                                           │  FAISS Index   │
                                           └──────┬────────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼────────┐
│  User Query  │────▶│  Retrieve K  │◀────│  Embed Query  │
└──────────────┘     │  chunks      │     └───────────────┘
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐     ┌───────────────┐
                     │  RAG Prompt  │◀────│  Web Search   │
                     │  Builder     │     │  (optional)   │
                     └──────┬───────┘     └───────────────┘
                            │
                     ┌──────▼───────┐
                     │  LLM Call    │──────▶  Answer
                     └──────────────┘
```

---

## Configuration Options

| Setting | Default | Location | Description |
|---|---|---|---|
| LLM Provider | OpenAI | Main panel | Choose OpenAI or HuggingFace Inference |
| Model | gpt-4o-mini | Main panel | The specific LLM model |
| Search Mode | Docs Only | Main panel | Documents / Web / Both |
| API Key | — | Main panel | Your LLM provider API key |
| Embedding Model | all-MiniLM-L6-v2 | Sidebar | Local Sentence-Transformer model |
| Chunk Size | 800 | Sidebar | Max characters per text chunk |
| Chunk Overlap | 200 | Sidebar | Overlap between consecutive chunks |
| Top-K | 4 | Sidebar | Number of chunks retrieved per question |
| Web Results | 5 | Sidebar | Number of web results to fetch |
| Temperature | 0.3 | Sidebar | LLM sampling temperature |
| Max Tokens | 1024 | Sidebar | Max tokens in the LLM response |

---

## Requirements

- Python 3.9+
- An API key for **OpenAI** or **HuggingFace** (for the LLM generation step)
- No GPU required — embeddings run on CPU by default

---

## License

MIT
