# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from collections.abc import AsyncIterator
from typing import List, Dict

# Third Party Library
import httpx
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Usage, Role, ImagePart
from relay.llm.errors import TimeoutError

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class GoogleLlm(BaseLlm):

    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        super().__init__("google", api_key, model_name)

    def _create_client(self, api_key: str):
        return genai.Client(api_key=api_key)

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=self._convert_messages(request.messages),
                config=self._build_config(request),
            )
        except genai_errors.APIError as e:
            # google-genai exposes the HTTP status on `.code`.
            raise self._error_for_status(getattr(e, "code", None), e) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(str(e)) from e

        structured = None
        if request.structured_output_schema is not None:
            # response_mime_type=application/json guarantees response.text is JSON.
            structured = self._parse_structured_text(response.text, request.structured_output_schema)

        return LlmResponse(
            content=response.text,
            model=self.model_name,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown",
            usage=Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                total_tokens=response.usage_metadata.total_token_count if response.usage_metadata else 0,
            ),
            structured=structured,
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

    def _convert_messages(self, messages: List[LlmMessage]) -> List[Dict]:
        # System messages handled via system_instruction in config - exclude here.
        return [
            {
                "role": "model" if msg.role == Role.assistant else "user",
                "parts": self._content_parts(msg),
            }
            for msg in messages if msg.role != Role.system
        ]

    def _content_parts(self, msg: LlmMessage) -> List[Dict]:
        # A plain-string message stays a single {"text": ...} part (backward-compat);
        # images become Gemini `inline_data` parts (base64 data + mime_type).
        parts: List[Dict] = []
        for part in self._parts(msg):
            if isinstance(part, ImagePart):
                parts.append({"inline_data": {"mime_type": part.mime_type, "data": part.data}})
            else:
                parts.append({"text": part.text})
        return parts

    def _build_config(self, request: LlmRequest) -> types.GenerateContentConfig:
        base = self._base_kwargs(request)
        config_kwargs = dict(
            temperature=base["temperature"],
            max_output_tokens=base["max_tokens"],
            top_p=request.top_p,
            top_k=request.top_k,
            system_instruction=request.system_prompt,
        )
        if request.structured_output_schema is not None:
            # Gemini's native structured-output path: JSON mime type + schema.
            # response_schema accepts a JSON-schema dict (OpenAPI subset).
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.structured_output_schema
        return types.GenerateContentConfig(**config_kwargs)

    async def list_models(self) -> List[str]:
        models = await self.client.aio.models.list()
        return sorted([m.name.replace("models/", "") for m in models])
