"""
tests.test_search
~~~~~~~~~~~~~~~~~

Unit tests for Google Serper search integration and keyword triggers.
Uses mocked HTTP responses to avoid live network calls.
"""

import httpx
import pytest
from answerkit.exceptions import WebSearchError
from answerkit.search import SerperClient, should_trigger_search


def test_should_trigger_search():
    assert should_trigger_search("What is the latest score today?") is True
    assert should_trigger_search("Current weather in London") is True
    assert should_trigger_search("Who is the CEO of Apple in 2025?") is True
    assert should_trigger_search("Explain how quicksort works") is False
    assert should_trigger_search("") is False


@pytest.mark.asyncio
async def test_serper_client_search_success():
    # Mock transport
    def mock_handler(request: httpx.Request):
        assert request.headers.get("X-API-KEY") == "test_serper_key"
        json_data = {
            "answerBox": {
                "title": "Quick Answer Box",
                "answer": "RAG enhances LLM outputs with external data.",
            },
            "organic": [
                {
                    "title": "What is Retrieval-Augmented Generation?",
                    "link": "https://aws.amazon.com/what-is/rag/",
                    "snippet": "Retrieval-Augmented Generation is the process of optimizing...",
                }
            ],
        }
        return httpx.Response(200, json=json_data)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = SerperClient(api_key="test_serper_key", http_client=mock_http)
        results = await client.search("What is RAG?")

        assert len(results) == 2
        assert results[0].title == "Quick Answer Box"
        assert "RAG enhances" in results[0].snippet
        assert results[1].title == "What is Retrieval-Augmented Generation?"
        assert results[1].link == "https://aws.amazon.com/what-is/rag/"


@pytest.mark.asyncio
async def test_serper_client_empty_query():
    client = SerperClient(api_key="test_key")
    results = await client.search("")
    assert results == []
    await client.close()


@pytest.mark.asyncio
async def test_serper_client_missing_key():
    with pytest.raises(WebSearchError, match="Serper API key is required"):
        SerperClient(api_key="")


@pytest.mark.asyncio
async def test_serper_client_http_error():
    def mock_handler(request: httpx.Request):
        return httpx.Response(403, text="Forbidden: Invalid API key")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = SerperClient(api_key="invalid_key", http_client=mock_http)
        with pytest.raises(WebSearchError, match="Serper API returned status 403"):
            await client.search("test query")
