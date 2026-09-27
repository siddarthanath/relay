# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import json
from collections.abc import AsyncIterator
from typing import List, Dict

# Third Party Library
import httpx
import anthropic
from anthropic import AsyncAnthropic

# Private Library
from relay.llm.base import BaseLlm, STRUCTURED_TOOL_NAME
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Usage, Role, ImagePart
from relay.llm.errors import TimeoutError, ProviderError, BadRequestError

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class AnthropicLlm(BaseLlm):
    """Anthropic Claude LLM provider implementation."""

    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        super().__init__("anthropic", api_key, model_name)

    def _create_client(self, api_key: str):
        return AsyncAnthropic(api_key=api_key)

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        try:
            response = await self.client.messages.create(**self._build_kwargs(request))
        except anthropic.APITimeoutError as e:
            raise TimeoutError(str(e)) from e
        except anthropic.APIStatusError as e:
            # 429/401/403/529/503/400 → typed error; anything else → ProviderError.
            raise self._error_for_status(e.status_code, e) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(str(e)) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(str(e)) from e

        structured = None
        if request.structured_output_schema is not None:
            # Structured output arrives as the forced tool's input (already a dict).
            tool_input = self._extract_tool_input(response)
            structured = self._validate_structured(tool_input, request.structured_output_schema)
            content = json.dumps(tool_input)
        else:
            content = response.content[0].text

        return LlmResponse(
            content=content,
            model=response.model,
            finish_reason=response.stop_reason,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            structured=structured,
        )

    @staticmethod
    def _extract_tool_input(response) -> dict:
        """Pull the input dict from the forced-tool-use block in the response."""
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input
        # Model failed to emit the tool call it was forced into — treat as malformed.
        raise BadRequestError("Structured output tool call was not present in the response")

    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._build_kwargs(request)) as stream:
            async for text in stream.text_stream:
                yield text

    def _convert_messages(self, messages: List[LlmMessage]) -> List[Dict]:
        # System messages handled via top-level system param - exclude here.
        converted: List[Dict] = []
        for msg in messages:
            if msg.role == Role.system:
                continue
            # Plain-string messages keep the original string-content shape (backward-compat);
            # multimodal messages emit Anthropic's native content-block list.
            if isinstance(msg.content, str):
                converted.append({"role": msg.role.value, "content": msg.content})
            else:
                converted.append({"role": msg.role.value, "content": self._content_blocks(msg)})
        return converted

    def _content_blocks(self, msg: LlmMessage) -> List[Dict]:
        blocks: List[Dict] = []
        for part in self._parts(msg):
            if isinstance(part, ImagePart):
                # Anthropic image block: base64 source with media_type + data.
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": part.mime_type, "data": part.data},
                })
            else:
                blocks.append({"type": "text", "text": part.text})
        return blocks

    def _build_kwargs(self, request: LlmRequest) -> Dict:
        kwargs = {
            **self._base_kwargs(request),
            "messages": self._convert_messages(request.messages),
            "max_tokens": request.max_tokens or 4096,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if request.structured_output_schema is not None:
            # Anthropic has no JSON-schema response mode, so we force a single tool call
            # whose input_schema *is* the requested schema and read the tool input back.
            kwargs["tools"] = [{
                "name": STRUCTURED_TOOL_NAME,
                "description": "Return the result strictly matching the provided JSON schema.",
                "input_schema": request.structured_output_schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
        return kwargs

    async def list_models(self) -> List[str]:
        # Anthropic's SDK has no official model-listing method yet, so fall back to a
        # direct httpx call against the /v1/models endpoint.
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": self.client.api_key, "anthropic-version": "2023-06-01"},
            )
            response.raise_for_status()
            return [m["id"] for m in response.json()["data"]]
