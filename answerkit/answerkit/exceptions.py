"""
answerkit.exceptions
~~~~~~~~~~~~~~~~~~~~

Custom exception hierarchy for the AnswerKit SDK.
Ensures clean, developer-friendly, and safe error messages without exposing credentials.
"""

from typing import Optional


class AnswerKitError(Exception):
    """Base exception for all errors raised by AnswerKit SDK."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


class ConfigurationError(AnswerKitError):
    """Raised when SDK configuration is missing or invalid."""
    pass


class APIKeyError(ConfigurationError):
    """Raised when an API key is missing or invalid."""
    pass


class ModelError(AnswerKitError):
    """Raised when LLM model execution or fallback cascade fails."""
    pass


class StreamingError(AnswerKitError):
    """Raised when an error occurs during streaming response generation."""
    pass


class WebSearchError(AnswerKitError):
    """Raised when web search querying or parsing fails."""
    pass


class PromptError(AnswerKitError):
    """Raised when prompt enhancement or processing encounters an error."""
    pass
