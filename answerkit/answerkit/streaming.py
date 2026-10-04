"""
answerkit.streaming
~~~~~~~~~~~~~~~~~~~

Framework-independent async streaming primitives and utilities for AnswerKit.
"""

from typing import AsyncIterator, List


async def collect_stream(stream: AsyncIterator[str]) -> str:
    """
    Utility helper that consumes an async token stream and aggregates it into a complete string.
    """
    chunks: List[str] = []
    async for token in stream:
        chunks.append(token)
    return "".join(chunks)
