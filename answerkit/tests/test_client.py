"""
tests.test_client
~~~~~~~~~~~~~~~~~

Comprehensive unit and integration tests for the public AnswerKit client.
All external Groq and Serper API calls are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from answerkit import AnswerKit
from answerkit.exceptions import ConfigurationError, ModelError
from answerkit.models import SearchResult


@pytest.fixture
def mock_groq_completion():
    def _make_response(text: str):
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp
    return _make_response


@pytest.mark.asyncio
async def test_ask_basic(mock_groq_completion):
    ai = AnswerKit(groq_api_key="gsk_test12345678")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_groq_completion("RAG combines retrieval with generative models.")
    )
    ai._agent._groq_client = mock_client

    answer = await ai.ask("What is RAG?")
    assert answer == "RAG combines retrieval with generative models."
    mock_client.chat.completions.create.assert_called_once()
    await ai.close()


@pytest.mark.asyncio
async def test_ask_with_conversation_history(mock_groq_completion):
    ai = AnswerKit(groq_api_key="gsk_test12345678")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_groq_completion("It is useful because it reduces hallucinations.")
    )
    ai._agent._groq_client = mock_client

    history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval Augmented Generation."},
    ]

    answer = await ai.ask("Why is it useful?", conversation_history=history)
    assert "reduces hallucinations" in answer

    # Verify messages passed to completion
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) == 4  # system prompt + 2 history + 1 current prompt
    assert messages[1]["content"] == "What is RAG?"
    assert messages[2]["content"] == "RAG stands for Retrieval Augmented Generation."
    assert messages[3]["content"] == "Why is it useful?"
    await ai.close()


@pytest.mark.asyncio
async def test_streaming():
    ai = AnswerKit(groq_api_key="gsk_test12345678")

    # Create mock chunk stream
    class MockChunk:
        def __init__(self, text):
            delta = MagicMock()
            delta.content = text
            choice = MagicMock()
            choice.delta = delta
            self.choices = [choice]

    async def mock_stream_gen():
        for chunk in ["Hello", " ", "World", "!"]:
            yield MockChunk(chunk)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream_gen())
    ai._agent._groq_client = mock_client

    collected = []
    async for token in ai.stream("Hi"):
        collected.append(token)

    assert "".join(collected) == "Hello World!"
    await ai.close()


@pytest.mark.asyncio
async def test_model_fallback_cascade(mock_groq_completion):
    ai = AnswerKit(
        groq_api_key="gsk_test12345678",
        model="primary-failed-model",
        fallback_models=["fallback-success-model"],
    )

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        model = kwargs.get("model")
        if model == "primary-failed-model":
            raise Exception("404 model_not_found: Model decommissioned")
        elif model == "fallback-success-model":
            return mock_groq_completion("Response from fallback model")
        raise Exception("Unknown model")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = mock_create
    ai._agent._groq_client = mock_client

    answer = await ai.ask("Test question")
    assert answer == "Response from fallback model"
    assert call_count == 2
    await ai.close()


@pytest.mark.asyncio
async def test_model_all_fail_raises_model_error():
    ai = AnswerKit(
        groq_api_key="gsk_test12345678",
        model="model1",
        fallback_models=["model2"],
    )

    async def mock_create(**kwargs):
        raise Exception("429 Rate limit exceeded")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = mock_create
    ai._agent._groq_client = mock_client

    with pytest.raises(ModelError, match="All candidate models failed"):
        await ai.ask("Test question")
    await ai.close()


@pytest.mark.asyncio
async def test_ask_missing_groq_key_raises_config_error():
    ai = AnswerKit(groq_api_key="")
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY is required"):
        await ai.ask("Hello")


@pytest.mark.asyncio
async def test_ask_with_web_search_metadata(mock_groq_completion):
    ai = AnswerKit(
        groq_api_key="gsk_test12345678",
        serper_api_key="serper_test",
        enable_web_search=True,
    )

    # Mock Serper search
    mock_search = AsyncMock(
        return_value=[
            SearchResult(
                title="Latest News Today",
                link="https://news.example.com",
                snippet="AI is advancing rapidly in 2025.",
            )
        ]
    )
    ai._search_client = AsyncMock()
    ai._search_client.search = mock_search

    # Mock Groq
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_groq_completion("Based on latest news, AI is advancing rapidly.")
    )
    ai._agent._groq_client = mock_client

    result = await ai.ask_with_metadata("What is the latest news today?")
    assert "advancing rapidly" in result.content
    assert result.used_search is True
    assert len(result.sources) == 1
    assert result.sources[0].title == "Latest News Today"
    await ai.close()


@pytest.mark.asyncio
async def test_polish_prompt(mock_groq_completion):
    ai = AnswerKit(groq_api_key="gsk_test12345678")

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=mock_groq_completion("Provide a comprehensive Python tutorial covering async/await fundamentals.")
    )
    ai._agent._groq_client = mock_client

    polished = await ai.polish_prompt("learn async python")
    assert "comprehensive Python tutorial" in polished
    await ai.close()


@pytest.mark.asyncio
async def test_context_manager(mock_groq_completion):
    async with AnswerKit(groq_api_key="gsk_test12345678") as ai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_groq_completion("Answer inside context manager")
        )
        ai._agent._groq_client = mock_client

        ans = await ai.ask("Hello")
        assert ans == "Answer inside context manager"
