"""
tests.test_prompts
~~~~~~~~~~~~~~~~~~

Unit tests for message building, prompt formatting, and search context injection.
"""

from answerkit.models import ChatMessage, SearchResult
from answerkit.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_chat_messages,
    format_search_context,
)


def test_format_search_context_empty():
    assert format_search_context([]) == ""


def test_format_search_context_with_results():
    results = [
        SearchResult(title="Result 1", link="https://res1.com", snippet="Snippet 1"),
        SearchResult(title="Result 2", link="https://res2.com", snippet="Snippet 2"),
    ]
    context = format_search_context(results)
    assert "[Live Google Search Context]:" in context
    assert "Title: Result 1" in context
    assert "https://res1.com" in context
    assert "Snippet 1" in context
    assert "Title: Result 2" in context


def test_build_chat_messages_simple():
    messages = build_chat_messages(prompt="What is Python?")
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "What is Python?"}


def test_build_chat_messages_with_history_and_search():
    history = [
        {"role": "user", "content": "Hi"},
        ChatMessage(role="assistant", content="Hello! How can I help?"),
    ]
    search_context = "[Search context here]"
    custom_sys = "You are a specialized math bot."

    messages = build_chat_messages(
        prompt="Solve 2+2",
        system_prompt=custom_sys,
        conversation_history=history,
        search_context=search_context,
    )

    assert len(messages) == 5
    assert messages[0] == {"role": "system", "content": custom_sys}
    assert messages[1] == {"role": "user", "content": "Hi"}
    assert messages[2] == {"role": "assistant", "content": "Hello! How can I help?"}
    assert messages[3] == {"role": "system", "content": "[Search context here]"}
    assert messages[4] == {"role": "user", "content": "Solve 2+2"}
