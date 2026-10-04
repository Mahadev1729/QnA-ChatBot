"""
answerkit.config
~~~~~~~~~~~~~~~~

Configuration management for AnswerKit SDK.
Supports explicit constructor options, environment variable fallbacks, and credential masking.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from answerkit.exceptions import ConfigurationError
from answerkit.models import DEFAULT_FALLBACK_MODELS, DEFAULT_MODEL


def _mask_secret(secret: Optional[str]) -> str:
    """Masks secret tokens for safe representation in logs/repr."""
    if not secret:
        return "<not-set>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


@dataclass
class AnswerKitConfig:
    """
    Configuration container for AnswerKit.

    Precedence:
      1. Explicit argument in constructor
      2. Environment variable (GROQ_API_KEY, SERPER_API_KEY, ANSWERKIT_MODEL, ANSWERKIT_ENABLE_WEB_SEARCH)
      3. Default SDK fallback value
    """
    groq_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    enable_web_search: bool = False
    fallback_models: List[str] = field(default_factory=lambda: list(DEFAULT_FALLBACK_MODELS))
    temperature: float = 0.3
    max_retries: int = 2
    timeout: float = 60.0
    system_prompt: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        groq_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_web_search: Optional[bool] = None,
        fallback_models: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> "AnswerKitConfig":
        """Constructs an AnswerKitConfig resolving environment variables."""
        # Resolve Groq API Key
        resolved_groq_key = (
            groq_api_key
            if groq_api_key is not None
            else os.getenv("GROQ_API_KEY")
        )

        # Resolve Serper API Key
        resolved_serper_key = (
            serper_api_key
            if serper_api_key is not None
            else os.getenv("SERPER_API_KEY")
        )

        # Resolve Model
        resolved_model = (
            model
            if model is not None
            else os.getenv("ANSWERKIT_MODEL")
            or os.getenv("GROQ_MODEL")
            or DEFAULT_MODEL
        )

        # Resolve Web Search Enablement
        if enable_web_search is not None:
            resolved_enable_search = bool(enable_web_search)
        else:
            env_search = os.getenv("ANSWERKIT_ENABLE_WEB_SEARCH", "").strip().lower()
            resolved_enable_search = env_search in ("1", "true", "yes", "on")

        # Resolve Fallback Models
        resolved_fallbacks = (
            list(fallback_models)
            if fallback_models is not None
            else list(DEFAULT_FALLBACK_MODELS)
        )

        # Resolve Temperature
        resolved_temp = temperature if temperature is not None else 0.3
        resolved_retries = max_retries if max_retries is not None else 2
        resolved_timeout = timeout if timeout is not None else 60.0

        return cls(
            groq_api_key=resolved_groq_key,
            serper_api_key=resolved_serper_key,
            model=resolved_model,
            enable_web_search=resolved_enable_search,
            fallback_models=resolved_fallbacks,
            temperature=resolved_temp,
            max_retries=resolved_retries,
            timeout=resolved_timeout,
            system_prompt=system_prompt,
        )

    def validate(self) -> None:
        """
        Validates configuration consistency.
        Raises ConfigurationError if required keys are missing.
        """
        if not self.groq_api_key or not self.groq_api_key.strip():
            raise ConfigurationError("GROQ_API_KEY is required.")

        if self.enable_web_search:
            if not self.serper_api_key or not self.serper_api_key.strip():
                raise ConfigurationError(
                    "SERPER_API_KEY is required when web search is enabled."
                )

    def __repr__(self) -> str:
        return (
            f"AnswerKitConfig("
            f"groq_api_key={_mask_secret(self.groq_api_key)!r}, "
            f"serper_api_key={_mask_secret(self.serper_api_key)!r}, "
            f"model={self.model!r}, "
            f"enable_web_search={self.enable_web_search!r}, "
            f"fallback_models={self.fallback_models!r}, "
            f"temperature={self.temperature!r})"
        )
