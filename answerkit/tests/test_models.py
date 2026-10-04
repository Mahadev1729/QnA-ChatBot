"""
tests.test_models
~~~~~~~~~~~~~~~~~

Unit tests for data models, type conversions, and model constants.
"""

from answerkit.models import (
    AnswerResult,
    ChatMessage,
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MODEL,
    POLISH_MODELS,
    SearchResult,
)


def test_chat_message():
    msg = ChatMessage(role="user", content="Hello AI")
    assert msg.role == "user"
    assert msg.content == "Hello AI"
    assert msg.to_dict() == {"role": "user", "content": "Hello AI"}

    from_dict_msg = ChatMessage.from_dict({"role": "assistant", "content": "Hello Human"})
    assert from_dict_msg.role == "assistant"
    assert from_dict_msg.content == "Hello Human"


def test_search_result():
    res = SearchResult(
        title="LangChain Documentation",
        link="https://python.langchain.com",
        snippet="Build context-aware applications.",
    )
    assert res.title == "LangChain Documentation"
    assert res.link == "https://python.langchain.com"
    assert res.to_dict() == {
        "title": "LangChain Documentation",
        "link": "https://python.langchain.com",
        "snippet": "Build context-aware applications.",
    }


def test_answer_result():
    res = AnswerResult(
        content="Artificial Intelligence is the simulation of human intelligence.",
        model="openai/gpt-oss-20b",
        used_search=True,
        sources=[
            SearchResult(title="AI Guide", link="https://example.com/ai", snippet="AI info")
        ],
    )
    assert res.content.startswith("Artificial Intelligence")
    assert res.model == "openai/gpt-oss-20b"
    assert res.used_search is True
    assert len(res.sources) == 1


def test_model_constants():
    assert isinstance(DEFAULT_MODEL, str)
    assert DEFAULT_MODEL in DEFAULT_FALLBACK_MODELS
    assert len(DEFAULT_FALLBACK_MODELS) >= 5
    assert len(POLISH_MODELS) >= 3
