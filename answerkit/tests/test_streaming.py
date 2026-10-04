"""
tests.test_streaming
~~~~~~~~~~~~~~~~~~~~

Unit tests for streaming iterator consumption and helpers.
"""

import pytest
from answerkit.streaming import collect_stream


@pytest.mark.asyncio
async def test_collect_stream():
    async def mock_token_generator():
        tokens = ["Hello", " ", "world", "!", " Welcome", " to", " AnswerKit."]
        for token in tokens:
            yield token

    aggregated = await collect_stream(mock_token_generator())
    assert aggregated == "Hello world! Welcome to AnswerKit."


@pytest.mark.asyncio
async def test_collect_empty_stream():
    async def empty_generator():
        if False:
            yield "never"

    aggregated = await collect_stream(empty_generator())
    assert aggregated == ""
