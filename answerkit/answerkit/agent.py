"""
answerkit.agent
~~~~~~~~~~~~~~~

Core orchestration engine for LLM generation, model fallback cascade,
streaming token iteration, and prompt polishing.
"""

from typing import AsyncIterator, List, Optional
import httpx
from groq import AsyncGroq

from answerkit.config import AnswerKitConfig
from answerkit.exceptions import ModelError, PromptError
from answerkit.models import POLISH_MODELS
from answerkit.prompts import POLISH_SYSTEM_PROMPT


class AgentEngine:
    """
    Internal execution engine that manages LLM calls, streaming, and model fallback.
    """

    def __init__(
        self,
        config: AnswerKitConfig,
        groq_client: Optional[AsyncGroq] = None,
    ) -> None:
        self.config = config
        self._groq_client = groq_client

    def _get_groq_client(self) -> AsyncGroq:
        """Returns initialized AsyncGroq client."""
        if self._groq_client is not None:
            return self._groq_client

        if not self.config.groq_api_key:
            raise ModelError("GROQ_API_KEY is not configured.")

        self._groq_client = AsyncGroq(
            api_key=self.config.groq_api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
        return self._groq_client

    def _get_candidate_models(self, preferred_model: Optional[str] = None) -> List[str]:
        """Builds ordered list of candidate models for fallback cascade."""
        candidates: List[str] = []
        if preferred_model:
            candidates.append(preferred_model)
        elif self.config.model:
            candidates.append(self.config.model)

        for fm in self.config.fallback_models:
            if fm not in candidates:
                candidates.append(fm)

        return candidates

    async def generate_response(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generates a full response with automatic model fallback cascade.
        """
        temp = temperature if temperature is not None else self.config.temperature
        candidate_models = self._get_candidate_models(model)
        client = self._get_groq_client()
        last_exception: Optional[Exception] = None

        for candidate in candidate_models:
            try:
                response = await client.chat.completions.create(
                    model=candidate,
                    messages=messages,  # type: ignore
                    temperature=temp,
                    stream=False,
                )
                choice = response.choices[0]
                content = choice.message.content
                if content is not None:
                    return content
                return ""
            except Exception as exc:
                err_str = str(exc).lower()
                # If model is unavailable, decommissioned, rate-limited, or not found, try next candidate
                is_fallback_error = any(
                    x in err_str
                    for x in [
                        "model_not_found",
                        "does not exist",
                        "404",
                        "decommissioned",
                        "not supported",
                        "400",
                        "429",
                        "rate_limit",
                        "service unavailable",
                        "503",
                    ]
                )
                last_exception = exc
                if not is_fallback_error and len(candidate_models) == 1:
                    raise ModelError(f"Model generation error on '{candidate}': {exc}") from exc

        raise ModelError(
            f"All candidate models failed ({', '.join(candidate_models)}). Last error: {last_exception}"
        ) from last_exception

    async def stream_response(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Streams response chunks with automatic model fallback cascade before first token.
        """
        temp = temperature if temperature is not None else self.config.temperature
        candidate_models = self._get_candidate_models(model)
        client = self._get_groq_client()
        last_exception: Optional[Exception] = None

        for candidate in candidate_models:
            try:
                response_stream = await client.chat.completions.create(
                    model=candidate,
                    messages=messages,  # type: ignore
                    temperature=temp,
                    stream=True,
                )

                async for chunk in response_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content

                # Completed successfully
                return

            except Exception as exc:
                err_str = str(exc).lower()
                is_fallback_error = any(
                    x in err_str
                    for x in [
                        "model_not_found",
                        "does not exist",
                        "404",
                        "decommissioned",
                        "not supported",
                        "400",
                        "429",
                        "rate_limit",
                        "service unavailable",
                        "503",
                    ]
                )
                last_exception = exc
                if not is_fallback_error and len(candidate_models) == 1:
                    raise ModelError(f"Model stream error on '{candidate}': {exc}") from exc

        raise ModelError(
            f"All candidate models failed to stream ({', '.join(candidate_models)}). Last error: {last_exception}"
        ) from last_exception

    async def polish_prompt_text(
        self,
        raw_prompt: str,
        preferred_model: Optional[str] = None,
    ) -> str:
        """
        Transforms a raw or brief user prompt into an enhanced, structured prompt.
        """
        if not raw_prompt or len(raw_prompt.strip()) < 2:
            return raw_prompt

        messages = [
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Enhance and clarify this prompt for maximum response quality:\n\n{raw_prompt.strip()}",
            },
        ]

        candidates = [preferred_model] if preferred_model else []
        for pm in POLISH_MODELS:
            if pm not in candidates:
                candidates.append(pm)

        client = self._get_groq_client()

        for candidate in candidates:
            try:
                response = await client.chat.completions.create(
                    model=candidate,
                    messages=messages,  # type: ignore
                    temperature=0.2,
                    stream=False,
                )
                choice = response.choices[0]
                text = (choice.message.content or "").strip().strip('"').strip("'")
                if text:
                    return text
            except Exception:
                continue

        # If polishing models all failed, return original raw prompt safely
        return raw_prompt.strip()

    async def close(self) -> None:
        """Closes Groq client session."""
        if self._groq_client is not None:
            await self._groq_client.close()
