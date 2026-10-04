"""
tests.test_config
~~~~~~~~~~~~~~~~~

Unit tests for AnswerKit configuration, validation, environment loading, and security masking.
"""

import pytest
from answerkit.config import AnswerKitConfig
from answerkit.exceptions import ConfigurationError


def test_config_explicit_initialization():
    config = AnswerKitConfig(
        groq_api_key="gsk_test1234567890",
        serper_api_key="serper_test123",
        model="llama-3.3-70b-versatile",
        enable_web_search=True,
        temperature=0.7,
    )
    assert config.groq_api_key == "gsk_test1234567890"
    assert config.serper_api_key == "serper_test123"
    assert config.model == "llama-3.3-70b-versatile"
    assert config.enable_web_search is True
    assert config.temperature == 0.7
    # Should not raise
    config.validate()


def test_config_missing_groq_key_raises():
    config = AnswerKitConfig(groq_api_key="")
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY is required"):
        config.validate()


def test_config_web_search_requires_serper_key():
    # When web search is enabled, serper key is required
    config = AnswerKitConfig(
        groq_api_key="gsk_valid_key",
        enable_web_search=True,
        serper_api_key=None,
    )
    with pytest.raises(ConfigurationError, match="SERPER_API_KEY is required"):
        config.validate()


def test_config_web_search_disabled_does_not_require_serper_key():
    config = AnswerKitConfig(
        groq_api_key="gsk_valid_key",
        enable_web_search=False,
        serper_api_key=None,
    )
    # Should validate successfully without Serper key
    config.validate()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env_gsk_12345678")
    monkeypatch.setenv("SERPER_API_KEY", "env_serper_456")
    monkeypatch.setenv("ANSWERKIT_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("ANSWERKIT_ENABLE_WEB_SEARCH", "true")

    config = AnswerKitConfig.from_env()
    assert config.groq_api_key == "env_gsk_12345678"
    assert config.serper_api_key == "env_serper_456"
    assert config.model == "openai/gpt-oss-120b"
    assert config.enable_web_search is True
    config.validate()


def test_config_constructor_overrides_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env_key")
    monkeypatch.setenv("ANSWERKIT_MODEL", "env_model")

    config = AnswerKitConfig.from_env(
        groq_api_key="override_key",
        model="override_model",
    )
    assert config.groq_api_key == "override_key"
    assert config.model == "override_model"


def test_config_secret_masking_in_repr():
    secret_key = "gsk_supersecretkey12345"
    config = AnswerKitConfig(
        groq_api_key=secret_key,
        serper_api_key="serper_secret_key_888",
    )
    repr_str = repr(config)
    assert secret_key not in repr_str
    assert "gsk_...2345" in repr_str
    assert "serp..._888" in repr_str
