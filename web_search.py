"""
Web search module.
Provides DuckDuckGo-based web search with a fallback HTML scraping method.
No API key required.
"""

import re
from typing import List
from urllib.parse import quote_plus

import requests


# ─────────────────────────────────────────────────────────────────────
# Primary Search (duckduckgo-search library)
# ─────────────────────────────────────────────────────────────────────
def web_search_duckduckgo(query: str, max_results: int = 5) -> List[dict]:
    """
    Search the web using the DuckDuckGo search library (no API key needed).
    Returns a list of {"title": ..., "url": ..., "snippet": ...} dicts.
    Falls back to HTML scraping if the library is unavailable.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except ImportError:
        return _web_search_fallback(query, max_results)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# Fallback Search (DuckDuckGo Lite HTML scraping)
# ─────────────────────────────────────────────────────────────────────
def _web_search_fallback(query: str, max_results: int = 5) -> List[dict]:
    """Fallback web search using DuckDuckGo Lite HTML scraping."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        results = []
        link_pattern = re.compile(
            r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        snippet_pattern = re.compile(
            r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
            re.DOTALL,
        )

        links = link_pattern.findall(resp.text)
        snippets = snippet_pattern.findall(resp.text)

        for i in range(min(len(links), len(snippets), max_results)):
            href, title = links[i]
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
            title = re.sub(r"<[^>]+>", "", title).strip()
            results.append({
                "title": title,
                "url": href,
                "snippet": snippet,
            })

        return results
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# Result Formatting
# ─────────────────────────────────────────────────────────────────────
def format_web_results_as_context(results: List[dict]) -> List[str]:
    """Convert web search results into context chunks for the LLM."""
    chunks = []
    for r in results:
        chunk = f"[Web: {r['title']}]\nURL: {r['url']}\n{r['snippet']}"
        chunks.append(chunk)
    return chunks
