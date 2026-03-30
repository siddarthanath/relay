# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import json
from collections.abc import AsyncIterator
from typing import List, Dict

# Third Party Library
import httpx

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #
 
class NativeRestOpenAILlm(BaseLlm):

    OPENAI_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        super().__init__("openai", api_key, model_name)

    def _create_client(self, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.OPENAI_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=60.0,
        )
 
    async def _generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.post("/chat/completions", json=self._build_body(request))
        response.raise_for_status()
        data = response.json()
 
        choice = data["choices"][0]
        usage  = data["usage"]
 
        return LlmResponse(
            content=choice["message"]["content"],
            model=data["model"],
            finish_reason=choice["finish_reason"],
            usage={
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        )
 
    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        body = {**self._build_body(request), "stream": True}
        async with self.client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
 
    def _convert_messages(self, messages: List[LlmMessage]) -> List[Dict[str, str]]:
        # OpenAI accepts system role natively - no exclusion needed
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
 
    def _build_body(self, request: LlmRequest) -> dict:
        base = self._base_kwargs(request)
        body = {
            "model": base["model"],
            "messages": self._build_messages(request),
            "temperature": base["temperature"],
            "max_tokens": base["max_tokens"],
        }
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            body["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            body["presence_penalty"] = request.presence_penalty
        return body
 
    def _build_messages(self, request: LlmRequest) -> List[Dict[str, str]]:
        messages = self._convert_messages(request.messages)
        if request.system_prompt:
            messages = [{"role": "system", "content": request.system_prompt}] + messages
        return messages
 
    async def list_models(self) -> List[str]:
        response = await self.client.get("/models")
        response.raise_for_status()
        return sorted([m["id"] for m in response.json()["data"]])