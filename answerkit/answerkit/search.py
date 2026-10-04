"""
answerkit.search
~~~~~~~~~~~~~~~~

Async Google Serper search integration and query detection.
"""

from typing import List, Optional
import httpx

from answerkit.exceptions import WebSearchError
from answerkit.models import SearchResult

SEARCH_TRIGGER_KEYWORDS: List[str] = [
    "today",
    "latest",
    "news",
    "current",
    "price",
    "weather",
    "release",
    "2024",
    "2025",
    "2026",
    "who is",
    "what is",
    "recent",
    "developments",
    "updates",
    "happening",
    "stock",
    "score",
]


def should_trigger_search(query: str) -> bool:
    """
    Determines if a query requires live web search based on keywords.
    """
    if not query:
        return False
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in SEARCH_TRIGGER_KEYWORDS)


class SerperClient:
    """
    Asynchronous Google Serper API client for live web search queries.
    """

    SERPER_ENDPOINT = "https://google.serper.dev/search"

    def __init__(
        self,
        api_key: str,
        timeout: float = 15.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise WebSearchError("Serper API key is required to initialize SerperClient.")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._external_client = http_client is not None
        self._client = http_client or httpx.AsyncClient(timeout=self.timeout)

    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Executes a web search on Google Serper and returns structured SearchResult objects.
        """
        if not query or not query.strip():
            return []

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query.strip(),
            "num": max(1, min(num_results, 10)),
        }

        try:
            response = await self._client.post(
                self.SERPER_ENDPOINT,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                raise WebSearchError(
                    f"Serper API returned status {response.status_code}: {response.text}"
                )

            data = response.json()
            results: List[SearchResult] = []

            # Check Answer Box
            if "answerBox" in data:
                ab = data["answerBox"]
                snippet = ab.get("answer") or ab.get("snippet") or ab.get("title", "")
                if snippet:
                    results.append(
                        SearchResult(
                            title=ab.get("title", "Quick Answer"),
                            link=ab.get("link", "https://google.com"),
                            snippet=str(snippet),
                        )
                    )

            # Check Knowledge Graph
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                snippet = kg.get("description") or kg.get("snippet", "")
                if snippet:
                    results.append(
                        SearchResult(
                            title=kg.get("title", "Knowledge Graph"),
                            link=kg.get("website") or kg.get("descriptionUrl", "https://google.com"),
                            snippet=str(snippet),
                        )
                    )

            # Check Organic Results
            for item in data.get("organic", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                    )
                )

            return results[:num_results]

        except WebSearchError:
            raise
        except Exception as exc:
            raise WebSearchError(f"Failed to execute Serper search: {exc}") from exc

    async def close(self) -> None:
        """Closes the underlying HTTP client if created internally."""
        if not self._external_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "SerperClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
