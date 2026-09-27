# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library

# Third Party Library
import pytest
from pydantic import ValidationError

# Private Library
from relay.llm.schemas import (
    LlmMessage,
    LlmRequest,
    LlmResponse,
    Role,
    Usage,
    TextPart,
    ImagePart,
)

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _message(**kwargs) -> LlmMessage:
    defaults = dict(role=Role.user, content="Hello")
    return LlmMessage(**(defaults | kwargs))


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[_message()])
    return LlmRequest(**(defaults | kwargs))


def _response(**kwargs) -> LlmResponse:
    defaults = dict(
        content="A response.",
        model="claude-opus-4-6",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    return LlmResponse(**(defaults | kwargs))


class TestRole:
    def test_is_str_enum(self):
        assert Role.user == "user"
        assert Role.assistant == "assistant"
        assert Role.system == "system"

    def test_user_value(self):
        assert Role.user.value == "user"

    def test_assistant_value(self):
        assert Role.assistant.value == "assistant"

    def test_system_value(self):
        assert Role.system.value == "system"

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            _message(role="unknown")


class TestLlmMessage:
    def test_valid_user_message(self):
        m = _message(role=Role.user, content="Hi")
        assert m.role == Role.user
        assert m.content == "Hi"

    def test_valid_assistant_message(self):
        m = _message(role=Role.assistant, content="Hello back")
        assert m.role == Role.assistant

    def test_valid_system_message(self):
        m = _message(role=Role.system, content="You are helpful.")
        assert m.role == Role.system

    def test_role_coercion_from_string(self):
        m = LlmMessage(role="user", content="Hi")
        assert m.role == Role.user

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            LlmMessage(content="Hi")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            LlmMessage(role=Role.user)


class TestContentParts:
    def test_str_content_backward_compat(self):
        m = _message(content="Hello")
        assert m.content == "Hello"

    def test_list_of_text_part(self):
        m = _message(content=[TextPart(text="Hi")])
        assert isinstance(m.content, list)
        assert m.content[0].text == "Hi"

    def test_image_part_fields(self):
        part = ImagePart(mime_type="image/png", data="YWJj")
        assert part.type == "image"
        assert part.mime_type == "image/png"
        assert part.data == "YWJj"

    def test_parts_discriminated_from_dicts(self):
        m = LlmMessage(
            role=Role.user,
            content=[{"type": "text", "text": "look"}, {"type": "image", "mime_type": "image/jpeg", "data": "AAAA"}],
        )
        assert isinstance(m.content[0], TextPart)
        assert isinstance(m.content[1], ImagePart)

    def test_invalid_part_type_raises(self):
        with pytest.raises(ValidationError):
            LlmMessage(role=Role.user, content=[{"type": "audio", "data": "x"}])


class TestUsage:
    def test_fields(self):
        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (1, 2, 3)

    def test_coerced_from_dict_on_response(self):
        r = _response()
        assert isinstance(r.usage, Usage)


class TestLlmRequest:
    def test_valid_minimal(self):
        r = _request()
        assert len(r.messages) == 1
        assert r.messages[0].role == Role.user

    def test_default_temperature(self):
        assert _request().temperature == 0.7

    def test_default_max_tokens_none(self):
        assert _request().max_tokens is None

    def test_default_system_prompt_none(self):
        assert _request().system_prompt is None

    def test_custom_temperature(self):
        r = _request(temperature=1.5)
        assert r.temperature == 1.5

    def test_temperature_zero_valid(self):
        assert _request(temperature=0.0).temperature == 0.0

    def test_temperature_two_valid(self):
        assert _request(temperature=2.0).temperature == 2.0

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            _request(temperature=-0.1)

    def test_temperature_above_two_raises(self):
        with pytest.raises(ValidationError):
            _request(temperature=2.1)

    def test_max_tokens_set(self):
        r = _request(max_tokens=512)
        assert r.max_tokens == 512

    def test_top_p_zero_valid(self):
        assert _request(top_p=0.0).top_p == 0.0

    def test_top_p_one_valid(self):
        assert _request(top_p=1.0).top_p == 1.0

    def test_top_p_above_one_raises(self):
        with pytest.raises(ValidationError):
            _request(top_p=1.1)

    def test_top_p_below_zero_raises(self):
        with pytest.raises(ValidationError):
            _request(top_p=-0.1)

    def test_top_k_one_valid(self):
        assert _request(top_k=1).top_k == 1

    def test_top_k_below_one_raises(self):
        with pytest.raises(ValidationError):
            _request(top_k=0)

    def test_frequency_penalty_bounds(self):
        assert _request(frequency_penalty=-2.0).frequency_penalty == -2.0
        assert _request(frequency_penalty=2.0).frequency_penalty == 2.0

    def test_frequency_penalty_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            _request(frequency_penalty=2.1)
        with pytest.raises(ValidationError):
            _request(frequency_penalty=-2.1)

    def test_presence_penalty_bounds(self):
        assert _request(presence_penalty=-2.0).presence_penalty == -2.0
        assert _request(presence_penalty=2.0).presence_penalty == 2.0

    def test_presence_penalty_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            _request(presence_penalty=2.1)
        with pytest.raises(ValidationError):
            _request(presence_penalty=-2.1)

    def test_system_prompt_set(self):
        r = _request(system_prompt="You are a helper.")
        assert r.system_prompt == "You are a helper."

    def test_multiple_messages(self):
        messages = [
            _message(role=Role.user, content="Hello"),
            _message(role=Role.assistant, content="Hi there"),
            _message(role=Role.user, content="How are you?"),
        ]
        r = _request(messages=messages)
        assert len(r.messages) == 3

    def test_missing_messages_raises(self):
        with pytest.raises(ValidationError):
            LlmRequest()


class TestLlmResponse:
    def test_valid(self):
        r = _response()
        assert r.content == "A response."
        assert r.model == "claude-opus-4-6"
        assert r.finish_reason == "stop"

    def test_usage_is_typed(self):
        r = _response()
        assert isinstance(r.usage, Usage)

    def test_usage_values(self):
        r = _response()
        assert r.usage.prompt_tokens == 10
        assert r.usage.completion_tokens == 5
        assert r.usage.total_tokens == 15

    def test_structured_defaults_none(self):
        assert _response().structured is None

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            LlmResponse(
                model="m",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            LlmResponse(
                content="hi",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
