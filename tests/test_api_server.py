"""Tests for the web API server: analyze + SSE observability events.

Uses httpx.ASGITransport so the app runs inside the same asyncio loop as
the test (matching uvicorn semantics); asserts the SSE stream carries the
fact-observability events: progress, claims, verification, ticker guard,
and completion.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from test_fact_checker import (  # noqa: E402
    _FakeAnalystLLM,
    _FakeTool,
    _FakeVerifyLLM,
)

import api_server  # noqa: E402
from api_server import app  # noqa: E402

TRANSPORT = httpx.ASGITransport(app=app)


def _make_fakes():
    analyst = _FakeAnalystLLM()
    verify = _FakeVerifyLLM([(True, "")])
    client = MagicMock()
    client.get_llm.side_effect = [verify, analyst]
    cross = {
        "fundamentals": [_FakeTool("tool_technical", "close 100.5")],
        "technical": [_FakeTool("tool_fundamental", "revenue 10e9")],
        "game_theory": [_FakeTool("tool_technical", "close 100.5")],
    }
    return analyst, verify, client, cross


def _patched_fakes():
    analyst, verify, client, cross = _make_fakes()
    return (
        patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client),
        patch("tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS", cross),
    ), analyst


async def _collect_stream(client: httpx.AsyncClient, job_id: str):
    events = []
    async with client.stream("GET", f"/api/v1/analyze/{job_id}/stream") as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


@pytest.mark.unit
class TestWebApi:
    def test_web_page_served(self):
        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.get("/")
                assert resp.status_code == 200
                assert "text/html" in resp.headers["content-type"]
                assert "TradingAgents" in resp.text
        asyncio.run(scenario())

    def test_analyze_streams_observability_events(self):
        patchers, _ = _patched_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "600519",
                    "date": "2026-08-20",
                    "analysts": ["fundamentals", "technical", "game_theory", "news_sentiment"],
                    "depth": 1,
                    "language": "Chinese",
                })
                assert resp.status_code == 200
                job_id = resp.json()["job_id"]
                events = await _collect_stream(client, job_id)

                kinds = [e["event"] for e in events]
                assert "started" in kinds
                assert "analysis_started" in kinds
                # Fact observability
                assert "claims" in kinds
                verifications = [e for e in events if e["event"] == "verification"]
                assert verifications, "verification events expected"
                for ev in verifications:
                    assert ev["passed"] is True
                    assert ev["attempts"] == 1
                    assert ev["items"], "verification event carries check items"
                # Ticker normalization surfaced by the guard
                assert any(
                    e["event"] == "ticker" and e["ticker"] == "600519.SH"
                    for e in events
                ), "ticker normalization event expected"
                # Progress + completion
                assert any(e["event"] == "agent_completed" for e in events)
                completed = [e for e in events if e["event"] == "completed"]
                assert completed and completed[0]["decision"] == "Buy"

                # Per-node completion events (fake LLMs emit no callbacks, so
                # llm_call events are covered by TestLLMRecordLinkage below)
                node_events = [e for e in events if e["event"] == "node"]
                assert node_events, "node completion events expected"
                assert any(e["node"] == "Analyst Team" for e in node_events)

        with patchers[0], patchers[1]:
            asyncio.run(scenario())

    def test_invalid_ticker_streams_error(self):
        patchers, _ = _patched_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "../evil", "date": "2026-08-20",
                    "analysts": ["fundamentals"], "depth": 1,
                })
                job_id = resp.json()["job_id"]
                events = await _collect_stream(client, job_id)
                kinds = [e["event"] for e in events]
                assert "error" in kinds

        with patchers[0], patchers[1]:
            asyncio.run(scenario())

    def test_result_endpoint_returns_verification(self):
        patchers, _ = _patched_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "600519.SH", "date": "2026-08-20",
                    "analysts": ["fundamentals"], "depth": 1,
                })
                job_id = resp.json()["job_id"]
                await _collect_stream(client, job_id)
                result = (await client.get(f"/api/v1/analyze/{job_id}/result")).json()
                assert result["status"] == "completed"
                assert result["decision"] == "Buy"
                assert result["verification"]["fundamentals"]["passed"] is True

        with patchers[0], patchers[1]:
            asyncio.run(scenario())


@pytest.mark.unit
class TestLLMRecordLinkage:
    """LLM input/output/token records link to streamed messages by run_id."""

    def _handler(self):
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        return api_server.SSEStatsHandler(loop, queue)

    def test_record_roundtrip_with_run_id(self):
        h = self._handler()
        h.on_chat_model_start({"name": "ChatOpenAI"}, "user: 分析 600519",
                              run_id="run-1")
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult
        msg = AIMessage(content="报告完成")
        msg.usage_metadata = {"input_tokens": 86, "output_tokens": 83}
        h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]),
                     run_id="run-1")

        rec = h.consume_record("run-1")
        assert rec is not None
        assert "分析 600519" in rec["input"]
        assert rec["output"] == "报告完成"
        assert rec["tokens"] == {"in": 86, "out": 83}
        assert rec["duration"] is not None
        # consumed exactly once
        assert h.consume_record("run-1") is None

    def test_fifo_fallback_without_run_id(self):
        h = self._handler()
        h.on_chat_model_start({"name": "ChatOpenAI"}, "input A")
        h.on_chat_model_start({"name": "ChatOpenAI"}, "input B")
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult
        for content, tokens in (("out A", (1, 2)), ("out B", (3, 4))):
            msg = AIMessage(content=content)
            msg.usage_metadata = {"input_tokens": tokens[0], "output_tokens": tokens[1]}
            h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
        r1 = h.consume_record(None)
        r2 = h.consume_record(None)
        assert "input A" in r1["input"] and r1["output"] == "out A"
        assert "input B" in r2["input"] and r2["output"] == "out B"
