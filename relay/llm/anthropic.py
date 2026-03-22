# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from collections.abc import AsyncIterator
from typing import List

# Third Party Library
from anthropic import AsyncAnthropic

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class NativeAnthropicLlm(BaseLlm):
    """Anthropic Claude LLM provider implementation."""

    def __init__(self, api_key: str, model_name: str = "claude-haiku-4.5") -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model_name: Claude model name (default: claude-haiku-4.5)

        """
        super().__init__(api_key, model_name)

    def _create_client(self, api_key: str):
        """Create and return the provider-specific API client instance."""
        return AsyncAnthropic(api_key=api_key)

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.messages.create(**self._build_kwargs(request))
        return LlmResponse(
            content=response.content[0].text,
            model=response.model,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens":     response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens":      response.usage.input_tokens + response.usage.output_tokens,
            },
        )
 
    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._build_kwargs(request)) as stream:
            async for text in stream.text_stream:
                yield text
 
    def _convert_messages(self, messages: List[LlmMessage]) -> list[dict]:
        # System messages handled via top-level system param — exclude here
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages if msg.role != Role.system
        ]
  
    def _build_kwargs(self, request: LlmRequest) -> dict:
        kwargs = {
            **self._base_kwargs(request),
            "messages": self._convert_messages(request.messages),
            "max_tokens": request.max_tokens or 4096,   # Anthropic requires max_tokens
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        return kwargs
