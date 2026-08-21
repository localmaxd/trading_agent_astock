"""Tests for the CLI live activity log (tool / LLM execution progress).

Every tool and LLM call is recorded as one activity entry that flips from
running to done (with duration + result summary) so the CLI UI can show the
pipeline's step-by-step progress in near-real time.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cli.stats_handler import StatsCallbackHandler


def _message_with_usage(input_tokens=86, output_tokens=83):
    msg = AIMessage(content="ok")
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return msg


def _llm_result(message):
    return LLMResult(generations=[[ChatGeneration(message=message)]])


@pytest.mark.unit
class TestToolActivity:
    def test_tool_start_end_pairs_into_one_done_entry(self):
        h = StatsCallbackHandler()
        h.on_tool_start({"name": "tool_fundamental"}, "ts_code=600519.SH")
        h.on_tool_end("财务指标: 毛利率 15.6%, ROIC 1.78%")

        stats = h.get_stats()
        assert stats["tool_calls"] == 1
        activity = h.get_activity()["tools"]
        assert len(activity) == 1
        entry = activity[0]
        assert entry["name"] == "tool_fundamental"
        assert entry["status"] == "done"
        assert "s" in entry["detail"]          # duration
        assert "ch" in entry["detail"]         # result size

    def test_running_entry_visible_before_end(self):
        h = StatsCallbackHandler()
        h.on_tool_start({"name": "tool_technical"}, "{}")
        activity = h.get_activity()["tools"]
        assert activity[0]["status"] == "running"
        assert activity[0]["args"]

    def test_tool_error_marks_failed(self):
        h = StatsCallbackHandler()
        h.on_tool_start({"name": "tool_news_sentiment"}, "{}")
        h.on_tool_error(RuntimeError("backend down"))
        assert h.get_activity()["tools"][0]["status"] == "error"

    def test_long_output_is_summarized(self):
        h = StatsCallbackHandler()
        h.on_tool_start({"name": "tool_fundamental"}, "{}")
        h.on_tool_end("x" * 5000)
        entry = h.get_activity()["tools"][0]
        assert entry["detail"]  # summarized, not the full 5000 chars


@pytest.mark.unit
class TestLLMActivity:
    def test_chat_model_start_end_pairs_and_counts_tokens(self):
        h = StatsCallbackHandler()
        h.on_chat_model_start({"name": "ChatOpenAI"}, [])
        h.on_llm_end(_llm_result(_message_with_usage(86, 83)))

        stats = h.get_stats()
        assert stats["llm_calls"] == 1
        assert stats["tokens_in"] == 86
        assert stats["tokens_out"] == 83

        entry = h.get_activity()["llms"][0]
        assert entry["status"] == "done"
        assert "86" in entry["detail"] and "83" in entry["detail"]

    def test_multiple_calls_update_the_newest_entry(self):
        h = StatsCallbackHandler()
        h.on_chat_model_start({"name": "ChatOpenAI"}, [])
        h.on_chat_model_start({"name": "ChatOpenAI"}, [])
        h.on_llm_end(_llm_result(_message_with_usage(10, 5)))
        entries = h.get_activity()["llms"]
        assert entries[0]["status"] == "running"  # first call still running
        assert entries[1]["status"] == "done"     # newest call completed
        assert h.get_stats()["tokens_in"] == 10
