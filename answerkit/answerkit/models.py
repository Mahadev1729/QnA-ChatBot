"""
answerkit.models
~~~~~~~~~~~~~~~~

Data models, model presets, and schema definitions for AnswerKit.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


DEFAULT_MODEL = "openai/gpt-oss-20b"

DEFAULT_FALLBACK_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "groq/compound",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

POLISH_MODELS: List[str] = [
    "groq/compound-mini",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


class ChatMessageDict(TypedDict):
    """Dictionary representation of a chat message."""
    role: str
    content: str


@dataclass
class ChatMessage:
    """Represents a conversation message."""
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(role=str(data.get("role", "user")), content=str(data.get("content", "")))


@dataclass
class SearchResult:
    """Represents a single search result from web search."""
    title: str
    link: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "link": self.link, "snippet": self.snippet}


@dataclass
class AnswerResult:
    """Structured result returned when full metadata is requested."""
    content: str
    model: str
    used_search: bool = False
    sources: List[SearchResult] = field(default_factory=list)
