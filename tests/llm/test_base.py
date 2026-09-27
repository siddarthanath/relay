# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library

# Private Library
from relay.llm.providers.anthropic import AnthropicLlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _llm() -> AnthropicLlm:
    with patch.object(AnthropicLlm, "_create_client", return_value=MagicMock()):
        return AnthropicLlm(api_key="fake-key", model_name="claude-opus-4-6")


class TestAsyncContextManager:
    def test_aenter_returns_self(self):
        llm = _llm()
        result = asyncio.run(llm.__aenter__())
        assert result is llm

    def test_aexit_calls_aclose(self):
        llm = _llm()
        llm._client.aclose = AsyncMock()
        asyncio.run(llm.__aexit__(None, None, None))
        llm._client.aclose.assert_called_once()

    def test_aexit_returns_false(self):
        llm = _llm()
        llm._client.aclose = AsyncMock()
        result = asyncio.run(llm.__aexit__(None, None, None))
        assert result is False

    def test_aexit_safe_when_no_aclose(self):
        llm = _llm()
        llm._client = object()  # plain object with no aclose
        asyncio.run(llm.__aexit__(None, None, None))  # should not raise

    def test_context_manager_closes_client(self):
        async def _run():
            with patch.object(AnthropicLlm, "_create_client", return_value=MagicMock()):
                llm = AnthropicLlm(api_key="fake-key", model_name="claude-opus-4-6")
                llm._client.aclose = AsyncMock()
                async with llm as ctx:
                    assert ctx is llm
                llm._client.aclose.assert_called_once()

        asyncio.run(_run())
