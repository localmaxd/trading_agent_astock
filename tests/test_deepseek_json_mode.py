"""Tests for the DeepSeek V4 json-mode structured output binding.

DeepSeek V4 thinking models reject explicit tool_choice (HTTP 400) and do
not support json_schema response_format, so the client falls back to
response_format=json_object + JsonOutputParser. These tests verify the
binding's behaviour without hitting the network; a separate smoke test
(tests/test_deepseek_json_mode.py::test_live_api_smoke, marked integration)
exercises the real API.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

import pytest

from tradingagents.llm_clients.openai_client import (
    DeepSeekChatOpenAI,
    _DeepSeekJsonModeLLM,
)


class _SampleRating(BaseModel):
    """Tiny schema used to exercise the binding."""

    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"] = Field(...)
    rationale: str = Field(default="")


class _FakeBound:
    """Fake runnable returned by llm.bind(...) in the tests."""

    def __init__(self, response: str):
        self.response = response

    def invoke(self, prompt, **kwargs):
        return AIMessage(content=self.response)


class _FakeLLMWithBind:
    """Minimal stand-in LLM exposing only bind() — avoids Pydantic model
    attribute restrictions when exercising _DeepSeekJsonModeLLM directly."""

    def __init__(self, response: str):
        self._bound = _FakeBound(response)

    def bind(self, **kwargs):
        assert kwargs.get("response_format") == {"type": "json_object"}
        return self._bound


def _make_llm(model: str) -> DeepSeekChatOpenAI:
    return DeepSeekChatOpenAI(
        model=model,
        api_key="sk-test",
        base_url="https://api.deepseek.com",
    )


@pytest.mark.unit
class TestDeepSeekJsonMode:
    def test_v4_model_returns_json_mode_binding(self):
        llm = _make_llm("deepseek-v4-flash")
        structured = llm.with_structured_output(_SampleRating)
        assert isinstance(structured, _DeepSeekJsonModeLLM)

    def test_v4_pro_also_uses_json_mode(self):
        llm = _make_llm("deepseek-v4-pro")
        assert isinstance(llm.with_structured_output(_SampleRating), _DeepSeekJsonModeLLM)

    def test_reasoner_still_raises_not_implemented(self):
        llm = _make_llm("deepseek-reasoner")
        with pytest.raises(NotImplementedError):
            llm.with_structured_output(_SampleRating)

    def test_non_v4_models_keep_function_calling_path(self, monkeypatch):
        llm = _make_llm("deepseek-chat")
        # If the base-class path is taken, with_structured_output returns a
        # runnable chain (RunnableBinding), NOT our json-mode wrapper.
        structured = llm.with_structured_output(_SampleRating)
        assert not isinstance(structured, _DeepSeekJsonModeLLM)

    def test_json_reply_is_parsed_into_pydantic(self):
        structured = _DeepSeekJsonModeLLM(
            _FakeLLMWithBind('{"rating": "Hold", "rationale": "balanced"}'),
            _SampleRating,
        )
        result = structured.invoke("some prompt")
        assert isinstance(result, _SampleRating)
        assert result.rating == "Hold"
        assert result.rationale == "balanced"

    def test_malformed_json_raises_parser_exception(self):
        structured = _DeepSeekJsonModeLLM(
            _FakeLLMWithBind("not json at all"),
            _SampleRating,
        )
        with pytest.raises(OutputParserException):
            structured.invoke("some prompt")

    def test_bind_uses_json_object_response_format(self):
        # Covered implicitly by _FakeLLMWithBind.bind asserting the format;
        # this test documents the contract.
        structured = _DeepSeekJsonModeLLM(
            _FakeLLMWithBind('{"rating": "Buy"}'),
            _SampleRating,
        )
        result = structured.invoke("x")
        assert result.rating == "Buy"
