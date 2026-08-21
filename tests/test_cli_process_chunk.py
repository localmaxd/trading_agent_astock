"""Tests for the CLI's live-update pipeline (process_chunk).

The CLI streams the graph in a background thread and refreshes the UI on a
timer; process_chunk runs on the UI thread and updates message_buffer.
These tests verify that chunk processing keeps agent statuses, report
sections, messages and tool calls in sync with the graph state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cli.main import MessageBuffer, process_chunk

ANALYSTS = ["fundamentals", "technical", "game_theory", "news_sentiment"]


def _make_buffer():
    buf = MessageBuffer()
    buf.init_for_analysis(ANALYSTS)
    return buf


def _message(msg_id="m1", content="hello", tool_calls=None):
    m = MagicMock()
    m.id = msg_id
    m.content = content
    m.tool_calls = tool_calls or []
    return m


@pytest.mark.unit
class TestProcessChunk:
    def test_analyst_report_updates_status_and_section(self):
        buf = _make_buffer()
        chunk = {"fundamentals_report": "ROIC 6%", "messages": []}
        process_chunk(buf, chunk)
        assert buf.report_sections["fundamentals_report"] == "ROIC 6%"
        assert buf.agent_status["Fundamentals Analyst (基本面)"] == "completed"

    def test_message_and_tool_calls_recorded_once(self):
        buf = _make_buffer()
        tool = MagicMock()
        tool.name = "tool_fundamental"
        tool.args = {"ts_code": "600519.SH"}
        chunk = {"messages": [_message(msg_id="m1", tool_calls=[tool])]}
        process_chunk(buf, chunk)
        process_chunk(buf, chunk)  # same msg id twice -> deduplicated
        assert len(buf.messages) == 1
        assert len(buf.tool_calls) == 1
        assert buf.tool_calls[0][1] == "tool_fundamental"

    def test_debate_updates_research_team(self):
        buf = _make_buffer()
        chunk = {
            "investment_debate_state": {
                "bull_history": "Bull says buy",
                "bear_history": "",
                "judge_decision": "",
            },
            "messages": [],
        }
        process_chunk(buf, chunk)
        assert buf.agent_status["Bull Researcher"] == "in_progress"
        assert "Bull Researcher Analysis" in buf.report_sections["investment_plan"]

        chunk["investment_debate_state"]["judge_decision"] = "Plan: buy 5%"
        chunk["investment_debate_state"]["bull_history"] = "Bull says buy"
        chunk["investment_debate_state"]["bear_history"] = "Bear says sell"
        process_chunk(buf, chunk)
        assert buf.agent_status["Research Manager"] == "completed"
        assert buf.agent_status["Trader"] == "in_progress"

    def test_trader_plan_advances_to_risk(self):
        buf = _make_buffer()
        chunk = {"trader_investment_plan": "Buy 200 shares", "messages": []}
        process_chunk(buf, chunk)
        assert buf.agent_status["Trader"] == "completed"
        assert buf.agent_status["Aggressive Analyst"] == "in_progress"
        assert buf.report_sections["trader_investment_plan"] == "Buy 200 shares"

    def test_risk_debate_completes_portfolio_manager(self):
        buf = _make_buffer()
        chunk = {
            "risk_debate_state": {
                "aggressive_history": "Go big",
                "conservative_history": "Go small",
                "neutral_history": "Middle",
                "judge_decision": "Final: Buy",
            },
            "messages": [],
        }
        process_chunk(buf, chunk)
        for agent in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"):
            assert buf.agent_status[agent] == "completed", agent
        assert "Portfolio Manager Decision" in buf.report_sections["final_trade_decision"]
