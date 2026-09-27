# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from collections.abc import AsyncIterator
from typing import List, Dict

# Third Party Library
import httpx
import openai
from openai import AsyncOpenAI

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Usage, ImagePart
from relay.llm.errors import TimeoutError, ProviderError

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class OpenAILlm(BaseLlm):
    """OpenAI LLM provider implementation."""

    STRUCTURED_FORMAT_NAME = "structured_output"

    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        super().__init__("openai", api_key, model_name)

    def _create_client(self, api_key: str):
        return AsyncOpenAI(api_key=api_key)

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        try:
            response = await self.client.chat.completions.create(
                **self._build_kwargs(request)
            )
        except openai.APITimeoutError as e:
            raise TimeoutError(str(e)) from e
        except openai.APIStatusError as e:
            raise self._error_for_status(e.status_code, e) from e
        except openai.APIConnectionError as e:
            raise ProviderError(str(e)) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(str(e)) from e

        content = response.choices[0].message.content
        structured = None
        if request.structured_output_schema is not None:
            # json_schema response_format returns the JSON as the message content.
            structured = self._parse_structured_text(content, request.structured_output_schema)

        return LlmResponse(
            content=content,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            structured=structured,
        )

    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            **self._build_kwargs(request, stream=True)
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _convert_messages(self, messages: List[LlmMessage]) -> List[Dict]:
        # OpenAI accepts system role natively - no exclusion needed.
        converted: List[Dict] = []
        for msg in messages:
            # Plain-string messages keep string content (backward-compat); multimodal
            # messages emit OpenAI's content-part list (text + image_url data URLs).
            if isinstance(msg.content, str):
                converted.append({"role": msg.role.value, "content": msg.content})
            else:
                converted.append({"role": msg.role.value, "content": self._content_parts(msg)})
        return converted

    def _content_parts(self, msg: LlmMessage) -> List[Dict]:
        parts: List[Dict] = []
        for part in self._parts(msg):
            if isinstance(part, ImagePart):
                # OpenAI takes images as a base64 data URL under image_url.
                data_url = f"data:{part.mime_type};base64,{part.data}"
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                parts.append({"type": "text", "text": part.text})
        return parts

    def _build_kwargs(self, request: LlmRequest, stream: bool = False) -> Dict:
        kwargs = {
            **self._base_kwargs(request),
            "messages": self._build_messages(request),
            "stream": stream,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.structured_output_schema is not None:
            # OpenAI's json_schema response_format. strict is left off so arbitrary
            # marking schemas (which may not set additionalProperties/required) are
            # accepted; Relay validates the response with jsonschema regardless.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.STRUCTURED_FORMAT_NAME,
                    "schema": request.structured_output_schema,
                },
            }
        return kwargs

    def _build_messages(self, request: LlmRequest) -> List[Dict]:
        messages = self._convert_messages(request.messages)
        if request.system_prompt:
            messages = [{"role": "system", "content": request.system_prompt}] + messages
        return messages

    async def list_models(self) -> List[str]:
        models = await self.client.models.list()
        return sorted([m.id for m in models.data])
