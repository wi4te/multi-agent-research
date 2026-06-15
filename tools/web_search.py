"""Web search tool (mock for now, can be replaced with Tavily/SerpAPI)."""
from typing import List, Dict


def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Mock web search. Replace with real API (Tavily, SerpAPI, etc.).
    """
    return [
        {
            "title": f"Result for: {query}",
            "url": "https://example.com",
            "snippet": f"Mock search result for query: {query}. Replace with real search API."
        }
    ]
