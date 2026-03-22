# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import json
from collections.abc import AsyncIterator
from typing import List, Dict

# Third Party Library
import httpx

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #
 
class RestAnthropicLlm(BaseLlm):
        
    ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
    ANTHROPIC_VERSION = "2023-06-01"
 
    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        super().__init__("anthropic", api_key, model_name)
 
    def _create_client(self, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.ANTHROPIC_BASE_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=60.0,
        )
 
    async def _generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.post("/messages", json=self._build_body(request))
        response.raise_for_status()
        data = response.json()
 
        return LlmResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            finish_reason=data["stop_reason"],
            usage={
                "prompt_tokens": data["usage"]["input_tokens"],
                "completion_tokens": data["usage"]["output_tokens"],
                "total_tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
            },
        )
 
    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        body = {**self._build_body(request), "stream": True}
        async with self.client.stream("POST", "/messages", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                if chunk.get("type") == "content_block_delta":
                    yield chunk["delta"].get("text", "")
 
    def _convert_messages(self, messages: List[LlmMessage]) -> List[Dict[str, str]]:
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages if msg.role != Role.system
        ]
 
    def _build_body(self, request: LlmRequest) -> Dict:
        base = self._base_kwargs(request)
        body = {
            "model": base["model"],
            "messages": self._convert_messages(request.messages),
            "temperature": base["temperature"],
            "max_tokens": base["max_tokens"] or 4096,  # Anthropic requires max_tokens
        }
        if request.system_prompt:
            body["system"] = request.system_prompt
        return body
 
    async def list_models(self) -> List[str]:
        response = await self.client.get("/models")
        response.raise_for_status()
        return sorted([m["id"] for m in response.json()["data"]])