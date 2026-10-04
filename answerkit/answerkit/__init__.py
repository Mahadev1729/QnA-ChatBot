"""
AnswerKit
~~~~~~~~~

A lightweight, standalone Python AI SDK for building Q&A applications with Groq,
async streaming, model fallback cascade, conversation context, optional Google Serper
web search, and prompt enhancement.
"""

from answerkit.client import AnswerKit
from answerkit.config import AnswerKitConfig
from answerkit.exceptions import (
    AnswerKitError,
    APIKeyError,
    ConfigurationError,
    ModelError,
    PromptError,
    StreamingError,
    WebSearchError,
)
from answerkit.models import (
    AnswerResult,
    ChatMessage,
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MODEL,
    SearchResult,
)
from answerkit.streaming import collect_stream

__version__ = "0.1.0"

__all__ = [
    "AnswerKit",
    "AnswerKitConfig",
    "AnswerKitError",
    "ConfigurationError",
    "APIKeyError",
    "ModelError",
    "StreamingError",
    "WebSearchError",
    "PromptError",
    "ChatMessage",
    "SearchResult",
    "AnswerResult",
    "DEFAULT_MODEL",
    "DEFAULT_FALLBACK_MODELS",
    "collect_stream",
    "__version__",
]
