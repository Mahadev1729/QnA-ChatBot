"""
answerkit.client
~~~~~~~~~~~~~~~~

Primary developer interface for the AnswerKit SDK.
Provides simple, async-first methods for Q&A, token streaming, prompt enhancement,
and web-augmented generation.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Union

from answerkit.agent import AgentEngine
from answerkit.config import AnswerKitConfig
from answerkit.exceptions import ConfigurationError
from answerkit.models import AnswerResult, ChatMessage, SearchResult
from answerkit.prompts import build_chat_messages, format_search_context
from answerkit.search import SerperClient, should_trigger_search


class AnswerKit:
    """
    Main client class for AnswerKit AI Q&A SDK.

    Examples:
        Basic Q&A:
        >>> from answerkit import AnswerKit
        >>> ai = AnswerKit(groq_api_key="YOUR_GROQ_API_KEY")
        >>> answer = await ai.ask("What is Retrieval Augmented Generation?")

        Streaming responses:
        >>> async for chunk in ai.stream("Explain transformers"):
        ...     print(chunk, end="", flush=True)

        Web-enabled answers:
        >>> ai = AnswerKit(
        ...     groq_api_key="YOUR_GROQ_API_KEY",
        ...     serper_api_key="YOUR_SERPER_API_KEY",
        ...     enable_web_search=True,
        ... )
        >>> answer = await ai.ask("What are the latest AI announcements today?")
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_web_search: Optional[bool] = None,
        fallback_models: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        config: Optional[AnswerKitConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an AnswerKit instance.

        Args:
            groq_api_key: Groq API key (defaults to GROQ_API_KEY environment variable).
            serper_api_key: Google Serper API key (defaults to SERPER_API_KEY environment variable).
            model: Primary LLM model identifier (defaults to ANSWERKIT_MODEL or 'openai/gpt-oss-20b').
            enable_web_search: Whether to enable Google Serper web search for queries.
            fallback_models: List of fallback models to try if primary model fails.
            temperature: LLM sampling temperature (default: 0.3).
            max_retries: Maximum number of request retries on temporary failures.
            timeout: Network timeout in seconds.
            system_prompt: Custom base system prompt for the assistant.
            config: Optional pre-constructed AnswerKitConfig instance.
        """
        if config is not None:
            self.config = config
        else:
            self.config = AnswerKitConfig.from_env(
                groq_api_key=groq_api_key,
                serper_api_key=serper_api_key,
                model=model,
                enable_web_search=enable_web_search,
                fallback_models=fallback_models,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                system_prompt=system_prompt,
            )

        self._agent = AgentEngine(self.config)
        self._search_client: Optional[SerperClient] = None

    def _get_search_client(self) -> SerperClient:
        """Returns or creates the SerperClient instance."""
        if not self.config.serper_api_key:
            raise ConfigurationError(
                "SERPER_API_KEY is required when web search is enabled."
            )
        if self._search_client is None:
            self._search_client = SerperClient(
                api_key=self.config.serper_api_key,
                timeout=self.config.timeout,
            )
        return self._search_client

    async def _resolve_search_context(
        self,
        prompt: str,
        enable_search_override: Optional[bool] = None,
    ) -> tuple[Optional[str], List[SearchResult], bool]:
        """Resolves web search if enabled and relevant."""
        is_search_enabled = (
            enable_search_override
            if enable_search_override is not None
            else self.config.enable_web_search
        )

        if not is_search_enabled:
            return None, [], False

        # Check if search is triggered by query keywords or forced via override
        if enable_search_override is True or should_trigger_search(prompt):
            client = self._get_search_client()
            try:
                results = await client.search(prompt)
                if results:
                    context = format_search_context(results)
                    return context, results, True
            except Exception as e:
                # If web search fails, we continue without interrupting the Q&A pipeline
                pass

        return None, [], False

    async def ask(
        self,
        prompt: str,
        conversation_history: Optional[List[Union[Dict[str, Any], ChatMessage]]] = None,
        model: Optional[str] = None,
        enable_web_search: Optional[bool] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Asks a question and returns the answer as a string.

        Args:
            prompt: User question or input prompt.
            conversation_history: Optional list of previous message dicts (e.g. `[{"role": "user", "content": "..."}]`).
            model: Optional model override for this single call.
            enable_web_search: Optional boolean to enable/disable web search for this call.
            temperature: Optional sampling temperature override.
            system_prompt: Optional custom system prompt override.

        Returns:
            The generated answer text as a string.

        Raises:
            ConfigurationError: If GROQ_API_KEY is not configured or SERPER_API_KEY is missing when search is enabled.
            ModelError: If all candidate models fail to generate an answer.
        """
        self.config.validate()

        search_context, _, _ = await self._resolve_search_context(
            prompt=prompt,
            enable_search_override=enable_web_search,
        )

        effective_system_prompt = (
            system_prompt
            if system_prompt is not None
            else self.config.system_prompt
        )

        messages = build_chat_messages(
            prompt=prompt,
            system_prompt=effective_system_prompt,
            conversation_history=conversation_history,
            search_context=search_context,
        )

        return await self._agent.generate_response(
            messages=messages,
            model=model,
            temperature=temperature,
        )

    async def ask_with_metadata(
        self,
        prompt: str,
        conversation_history: Optional[List[Union[Dict[str, Any], ChatMessage]]] = None,
        model: Optional[str] = None,
        enable_web_search: Optional[bool] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> AnswerResult:
        """
        Asks a question and returns structured metadata including sources and model used.
        """
        self.config.validate()

        search_context, sources, used_search = await self._resolve_search_context(
            prompt=prompt,
            enable_search_override=enable_web_search,
        )

        effective_system_prompt = (
            system_prompt
            if system_prompt is not None
            else self.config.system_prompt
        )

        messages = build_chat_messages(
            prompt=prompt,
            system_prompt=effective_system_prompt,
            conversation_history=conversation_history,
            search_context=search_context,
        )

        content = await self._agent.generate_response(
            messages=messages,
            model=model,
            temperature=temperature,
        )

        used_model = model or self.config.model
        return AnswerResult(
            content=content,
            model=used_model,
            used_search=used_search,
            sources=sources,
        )

    async def stream(
        self,
        prompt: str,
        conversation_history: Optional[List[Union[Dict[str, Any], ChatMessage]]] = None,
        model: Optional[str] = None,
        enable_web_search: Optional[bool] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Streams response chunks asynchronously token-by-token.

        Args:
            prompt: User question or input prompt.
            conversation_history: Optional list of previous conversation messages.
            model: Optional model override.
            enable_web_search: Optional search toggle.
            temperature: Optional sampling temperature override.
            system_prompt: Optional custom system prompt override.

        Yields:
            str: Each token/text chunk as it arrives from the model.
        """
        self.config.validate()

        search_context, _, _ = await self._resolve_search_context(
            prompt=prompt,
            enable_search_override=enable_web_search,
        )

        effective_system_prompt = (
            system_prompt
            if system_prompt is not None
            else self.config.system_prompt
        )

        messages = build_chat_messages(
            prompt=prompt,
            system_prompt=effective_system_prompt,
            conversation_history=conversation_history,
            search_context=search_context,
        )

        async for chunk in self._agent.stream_response(
            messages=messages,
            model=model,
            temperature=temperature,
        ):
            yield chunk

    async def polish_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Enhances and refines a user prompt into an optimal, high-quality prompt.

        Args:
            prompt: The draft or brief prompt to polish.
            model: Optional model override to use for polishing.

        Returns:
            The improved prompt text.
        """
        if not self.config.groq_api_key or not self.config.groq_api_key.strip():
            raise ConfigurationError("GROQ_API_KEY is required.")

        return await self._agent.polish_prompt_text(
            raw_prompt=prompt,
            preferred_model=model,
        )

    async def close(self) -> None:
        """Closes all underlying network sessions."""
        if self._search_client is not None:
            await self._search_client.close()
        await self._agent.close()

    async def __aenter__(self) -> "AnswerKit":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"AnswerKit(config={self.config!r})"
