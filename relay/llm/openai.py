# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from collections.abc import AsyncIterator
from typing import List

# Third Party Library
from openai import AsyncOpenAI

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class NativeOpenAILlm(BaseLlm):
    """OpenAI LLM provider implementation."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini") -> None:
        """Initialise OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: OpenAI model name (default: gpt-4o-mini)

        """
        super().__init__(api_key, model_name)

    def _create_client(self, api_key: str):
        """Create and return the provider-specific API client instance."""
        return AsyncOpenAI(api_key=api_key)

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.chat.completions.create(
            **self._build_kwargs(request)
        )
        return LlmResponse(
            content=response.choices[0].message.content,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            usage={
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            },
        )
 
    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            **self._build_kwargs(request, stream=True)
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
 
    def _convert_messages(self, messages: List[LlmMessage]) -> list[dict]:
        # OpenAI accepts system role natively — no exclusion needed
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
  
    def _build_kwargs(self, request: LlmRequest, stream: bool = False) -> dict:
        kwargs = {
            **self._base_kwargs(request),
            "messages": self._build_messages(request),
            "stream":   stream,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        return kwargs
 
    def _build_messages(self, request: LlmRequest) -> list[dict]:
        """Prepend system_prompt as a system message if provided."""
        messages = self._convert_messages(request.messages)
        if request.system_prompt:
            messages = [{"role": "system", "content": request.system_prompt}] + messages
        return messages
