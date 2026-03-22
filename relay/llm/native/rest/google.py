# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import json
from typing import List, Dict
from collections.abc import AsyncIterator

# Third Party Library
import httpx

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

class RestGoogleLlm(BaseLlm):

    GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        super().__init__("google", api_key, model_name)

    def _create_client(self, api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.GOOGLE_BASE_URL,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )

    async def _generate(self, request: LlmRequest) -> LlmResponse:
        response = await self.client.post(
            f"/{self.model_name}:generateContent",
            json=self._build_body(request),
        )
        response.raise_for_status()
        data = response.json()

        candidate = data["candidates"][0]
        usage = data.get("usageMetadata", {})

        return LlmResponse(
            content=candidate["content"]["parts"][0]["text"],
            model=self.model_name,
            finish_reason=candidate.get("finishReason", "unknown"),
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        )

    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            f"/{self.model_name}:streamGenerateContent",
            json=self._build_body(request),
            params={"alt": "sse"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:].strip())
                try:
                    yield chunk["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    continue

    def _convert_messages(self, messages: List[LlmMessage]) -> List[Dict]:
        return [
            {
                "role": "model" if msg.role == Role.assistant else "user",
                "parts": [{"text": msg.content}],
            }
            for msg in messages if msg.role != Role.system
        ]

    def _build_body(self, request: LlmRequest) -> Dict:
        base = self._base_kwargs(request)
        body = {
            "contents": self._convert_messages(request.messages),
            "generationConfig": {
                "temperature": base["temperature"],
                "maxOutputTokens": base["max_tokens"],
                "topP": request.top_p,
                "topK": request.top_k,
            },
        }
        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        return body

    async def list_models(self) -> List[str]:
        response = await self.client.get("/", params={"key": self.api_key})
        response.raise_for_status()
        models = response.json().get("models", [])
        return sorted([m["name"].replace("models/", "") for m in models])