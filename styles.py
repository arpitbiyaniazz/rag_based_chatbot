"""
Custom CSS styles for the Streamlit UI.
Returns the full <style> block as a string.
"""


def get_custom_css() -> str:
    """Return the complete custom CSS for the RAG chatbot UI."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0;
    }

    /* Config card (main area) */
    .config-card {
        background: linear-gradient(135deg, rgba(30, 30, 63, 0.6) 0%, rgba(25, 25, 50, 0.6) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .config-card h4 {
        color: #c4b5fd;
        margin: 0 0 0.6rem;
        font-size: 0.95rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131328 0%, #1e1e3f 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(168, 85, 247, 0.12) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px 16px 4px 16px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #e2e8f0;
    }
    .chat-assistant {
        background: linear-gradient(135deg, rgba(30, 30, 63, 0.7) 0%, rgba(25, 25, 50, 0.7) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px 16px 16px 4px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #cbd5e1;
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 99px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        color: #a5b4fc;
        margin: 0.15rem 0.2rem;
    }
    .source-pill-web {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 99px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        color: #6ee7b7;
        margin: 0.15rem 0.2rem;
    }
    .source-pill-web a {
        color: #6ee7b7;
        text-decoration: none;
    }
    .source-pill-web a:hover {
        text-decoration: underline;
    }

    /* Web result cards */
    .web-result {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .web-result .web-title {
        color: #6ee7b7;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .web-result .web-url {
        color: #94a3b8;
        font-size: 0.75rem;
        word-break: break-all;
    }
    .web-result .web-snippet {
        color: #cbd5e1;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* File uploader area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    /* Mode selector badges */
    .mode-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .mode-docs { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
    .mode-web { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
    .mode-both { background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(16, 185, 129, 0.2));
                  color: #c4b5fd; border: 1px solid rgba(99, 102, 241, 0.3); }

    /* Status indicator */
    .status-connected {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 99px; padding: 0.25rem 0.75rem; font-size: 0.8rem; color: #6ee7b7;
    }
    .status-disconnected {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 99px; padding: 0.25rem 0.75rem; font-size: 0.8rem; color: #fca5a5;
    }
    .status-dot-green { width: 8px; height: 8px; border-radius: 50%; background: #10b981;
                        display: inline-block; animation: pulse-green 2s infinite; }
    .status-dot-red   { width: 8px; height: 8px; border-radius: 50%; background: #ef4444;
                        display: inline-block; }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    </style>
    """
