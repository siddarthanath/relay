# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from collections.abc import AsyncIterator
from typing import List

# Third Party Library
from google import genai
from google.genai import types

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class NativeGoogleLlm(BaseLlm):

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        super().__init__(api_key, model_name)

    def _create_client(self, api_key: str):
        """Create and return the provider-specific API client instance."""
        return genai.Client(api_key=api_key)

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=self._convert_messages(request.messages),
            config=self._build_config(request),
        )
        return LlmResponse(
            content=response.text,
            model=self.model_name,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown",
            usage={
                "prompt_tokens":     response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                "total_tokens":      response.usage_metadata.total_token_count if response.usage_metadata else 0,
            },
        )
 
    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=self._convert_messages(request.messages),
            config=self._build_config(request),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
 
    def _convert_messages(self, messages: List[LlmMessage]) -> list[types.Content]:
        # System messages handled via system_instruction in config — exclude here
        return [
            types.Content(
                role="model" if msg.role == Role.assistant else "user",
                parts=[types.Part(text=msg.content)],
            )
            for msg in messages if msg.role != Role.system
        ]
  
    def _build_config(self, request: LlmRequest) -> types.GenerateContentConfig:
        base = self._base_kwargs(request)
        return types.GenerateContentConfig(
            temperature=base["temperature"],
            max_output_tokens=base["max_tokens"],
            top_p=request.top_p,
            top_k=request.top_k,
            system_instruction=request.system_prompt,
        )