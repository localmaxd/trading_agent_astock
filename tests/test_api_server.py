"""Tests for the web API server: analyze + SSE observability events.

Uses httpx.ASGITransport so the app runs inside the same asyncio loop as
the test (matching uvicorn semantics); asserts the SSE stream carries the
fact-observability events: progress, claims, verification, ticker guard,
and completion. Run-control tests cover pause (checkpoint) / resume /
stop (clear checkpoint).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
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
from tradingagents.errors import AnalysisStopped  # noqa: E402

TRANSPORT = httpx.ASGITransport(app=app)


@pytest.fixture(scope="module", autouse=True)
def _checkpoint_dir_env(tmp_path_factory):
    """Redirect the job checkpointer to a temp dir (default ~/.tradingagents)."""
    check_dir = tmp_path_factory.mktemp("web-checkpoints")
    os.environ["WEB_CHECKPOINT_DIR"] = str(check_dir)
    yield str(check_dir)
    os.environ.pop("WEB_CHECKPOINT_DIR", None)


class _SlowAnalystLLM(_FakeAnalystLLM):
    """Slow stand-in analyst LLM so pause/resume can land mid-run."""

    def _respond(self, prompt=None):
        time.sleep(0.25)
        return super()._respond(prompt)


class _BlockingAnalystLLM(_SlowAnalystLLM):
    """Counting slow LLM that honors a stop flag at every call boundary.

    真实客户端（ChatOpenAI 等）在停止时会由 SSEStatsHandler 的
    on_chat_model_start 抛出 AnalysisStopped；假 LLM 没有 LangChain 聊天模型
    回调，所以在 _respond 边界做同样的检查，模拟停止信号下沉。
    """

    def __init__(self, *args, should_stop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep = 0.6
        self.should_stop = should_stop or (lambda: False)

    def _respond(self, prompt=None):
        if self.should_stop():
            from tradingagents.errors import AnalysisStopped
            raise AnalysisStopped("analysis stopped by user")
        time.sleep(self.sleep)
        return super()._respond(prompt)


def _make_fakes():
    analyst = _FakeAnalystLLM()
    verify = _FakeVerifyLLM([(True, "")])
    client = MagicMock()
    client.get_llm.side_effect = [verify, analyst]
    cross = {
        "fundamentals": [_FakeTool("tool_fundamental", '{"roic": 6.0}')],
        "technical": [_FakeTool("tool_technical", '{"roic": 6.0}')],
        "game_theory": [_FakeTool("tool_game_theory", '{"roic": 6.0}')],
    }
    return analyst, verify, client, cross


def _patched_fakes():
    analyst, verify, client, cross = _make_fakes()
    return (
        patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client),
        patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", cross),
    ), analyst


def _patched_slow_fakes():
    """Same fakes but with a slow analyst LLM (pause/resume can land mid-run)."""
    analyst = _SlowAnalystLLM()
    verify = _FakeVerifyLLM([(True, "")])
    client = MagicMock()
    client.get_llm.side_effect = [verify, analyst]
    cross = {
        "fundamentals": [_FakeTool("tool_fundamental", '{"roic": 6.0}')],
        "technical": [_FakeTool("tool_technical", '{"roic": 6.0}')],
        "game_theory": [_FakeTool("tool_game_theory", '{"roic": 6.0}')],
    }
    return (
        patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client),
        patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", cross),
    ), analyst


def _patched_blocking_fakes():
    """Slow counting LLM so stop-interrupts-run can be asserted deterministically."""
    stop_event = threading.Event()
    analyst = _BlockingAnalystLLM(should_stop=stop_event.is_set)
    verify = _FakeVerifyLLM([(True, "")])
    client = MagicMock()
    client.get_llm.side_effect = [verify, analyst]
    cross = {
        "fundamentals": [_FakeTool("tool_fundamental", '{"roic": 6.0}')],
        "technical": [_FakeTool("tool_technical", '{"roic": 6.0}')],
        "game_theory": [_FakeTool("tool_game_theory", '{"roic": 6.0}')],
    }
    return (
        patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client),
        patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", cross),
    ), analyst, stop_event


async def _collect_stream(client: httpx.AsyncClient, job_id: str):
    events = []
    async with client.stream("GET", f"/api/v1/analyze/{job_id}/stream") as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


async def _wait_status(client: httpx.AsyncClient, job_id: str, targets, timeout=90.0):
    """Poll the result endpoint until the job reaches one of ``targets``."""
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        res = (await client.get(f"/api/v1/analyze/{job_id}/result")).json()
        status = res["status"]
        if status in targets:
            return res
        if status == "failed":
            raise AssertionError(f"job failed: {res.get('error')}")
        await asyncio.sleep(0.2)
    raise TimeoutError(f"job stuck in status {status!r}")


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
                    "analysts": ["fundamentals", "macro"],
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
                # Node lifecycle events drive the directed-graph flow animation:
                # task start (node_started) before completion (node), with the
                # exact path into subgraph internals.
                started = [e for e in events if e["event"] == "node_started"]
                assert started, "node_started events expected for the graph animation"
                assert any(e["node"] == "Analyst Team" for e in started)
                assert any(
                    e["node"] == "Analyst Team/Fundamentals Analyst/Analyze"
                    for e in started
                ), "nested analyst-subgraph node paths expected"

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


    def test_graph_endpoint_returns_directed_structure(self):
        patchers, _ = _patched_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.get("/api/v1/graph", params={
                    "analysts": "fundamentals,technical,game_theory,news_sentiment",
                })
                assert resp.status_code == 200
                g = resp.json()
                parent_ids = [n["id"] for n in g["parent"]["nodes"]]
                # parent chain: start -> guards -> stages -> end
                assert "__start__" in parent_ids and "__end__" in parent_ids
                assert "TickerGuard-Analyst Team" in parent_ids
                assert "Analyst Team" in parent_ids
                assert g["parent"]["edges"], "parent edges expected"
                # every stage carries its internal nodes + directed edges
                stages = {s["id"]: s for s in g["stages"]}
                assert "Analyst Team" in stages
                inner_ids = [n["id"] for n in stages["Analyst Team"]["nodes"]]
                assert "Fundamentals Analyst" in inner_ids
                fundamentals = next(
                    n for n in stages["Analyst Team"]["nodes"]
                    if n["id"] == "Fundamentals Analyst"
                )
                assert fundamentals["kind"] == "subgraph"
                branch_ids = [n["id"] for n in fundamentals["subgraph"]["nodes"]]
                assert "Analyze" in branch_ids
                assert "tools_fundamentals" in branch_ids
                assert "FactChecker-Fundamentals" in branch_ids
                assert stages["Analyst Team"]["edges"], "internal edges expected"
                # Conditional tool/fact-check routes now live inside the
                # independently compiled analyst branch.
                cond = [
                    e for e in fundamentals["subgraph"]["edges"]
                    if e["conditional"]
                ]
                assert cond, "conditional (router) edges expected"
                # guard nodes classified for the UI
                guard = next(n for n in g["parent"]["nodes"] if n["id"] == "TickerGuard-Analyst Team")
                assert guard["kind"] == "guard"
                stage = next(n for n in g["parent"]["nodes"] if n["id"] == "Analyst Team")
                assert stage["kind"] == "stage"

        with patchers[0], patchers[1]:
            asyncio.run(scenario())

    def test_graph_endpoint_filters_analysts(self):
        patchers, _ = _patched_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.get("/api/v1/graph", params={"analysts": "fundamentals"})
                assert resp.status_code == 200
                g = resp.json()
                stages = {s["id"]: s for s in g["stages"]}
                inner_ids = [n["id"] for n in stages["Analyst Team"]["nodes"]]
                assert "Fundamentals Analyst" in inner_ids
                assert "Technical Analyst" not in inner_ids

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
                              run_id="run-1", metadata={
                                  "langgraph_checkpoint_ns": "Analyst Team:abc",
                                  "langgraph_node": "Fundamentals Analyst",
                              })
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
        assert rec["node"] == "Analyst Team/Fundamentals Analyst"
        assert rec["path"] == ["Analyst Team", "Fundamentals Analyst"]
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


@pytest.mark.unit
class TestRunControl:
    """pause (checkpoint) / resume / stop (clear checkpoint) run control."""

    def test_pause_then_resume_completes(self):
        """pause -> checkpoint written -> resume completes from checkpoint."""
        from tradingagents.graph.checkpointer import has_checkpoint

        patchers, _ = _patched_slow_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "600519.SH",
                    "date": "2026-08-20",
                    "analysts": ["fundamentals"],
                    "depth": 1,
                    "language": "Chinese",
                })
                job_id = resp.json()["job_id"]
                # Pause immediately: the flag lands before the first chunk,
                # so the run stops at the first super-step boundary.
                r = await client.post(f"/api/v1/analyze/{job_id}/pause")
                assert r.status_code == 200
                res = await _wait_status(client, job_id, {"paused"}, timeout=30)
                assert res["status"] == "paused"
                # 暂停时进度已写入 checkpointer
                assert has_checkpoint(os.environ["WEB_CHECKPOINT_DIR"], "600519.SH", "2026-08-20")

                # Resume from the checkpoint: the run continues and completes
                r = await client.post(f"/api/v1/analyze/{job_id}/resume")
                assert r.status_code == 200
                res = await _wait_status(client, job_id, {"completed"}, timeout=90)
                assert res["status"] == "completed"
                assert res["decision"] == "Buy"
                # 完成后检查点被清空
                assert not has_checkpoint(os.environ["WEB_CHECKPOINT_DIR"], "600519.SH", "2026-08-20")

        with patchers[0], patchers[1]:
            asyncio.run(scenario())

    def test_stop_clears_checkpoint_and_marks_stopped(self):
        """stop (running/paused) interrupts the run and clears the checkpoint."""
        from tradingagents.graph.checkpointer import has_checkpoint

        patchers, _ = _patched_slow_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "600519.SH",
                    "date": "2026-08-20",
                    "analysts": ["fundamentals"],
                    "depth": 1,
                    "language": "Chinese",
                })
                job_id = resp.json()["job_id"]
                await client.post(f"/api/v1/analyze/{job_id}/pause")
                await _wait_status(client, job_id, {"paused"}, timeout=30)
                assert has_checkpoint(os.environ["WEB_CHECKPOINT_DIR"], "600519.SH", "2026-08-20")

                r = await client.post(f"/api/v1/analyze/{job_id}/stop")
                assert r.status_code == 200
                res = await _wait_status(client, job_id, {"stopped"}, timeout=30)
                assert res["status"] == "stopped"
                # 停止后 checkpointer 被清空
                assert not has_checkpoint(os.environ["WEB_CHECKPOINT_DIR"], "600519.SH", "2026-08-20")

        with patchers[0], patchers[1]:
            asyncio.run(scenario())

    def test_control_endpoints_reject_wrong_status(self):
        patchers, _ = _patched_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "600519.SH",
                    "date": "2026-08-20",
                    "analysts": ["fundamentals"],
                    "depth": 1,
                    "language": "Chinese",
                })
                job_id = resp.json()["job_id"]
                await _wait_status(client, job_id, {"completed"}, timeout=60)
                # 已完成的任务不可暂停/恢复/停止
                assert (await client.post(f"/api/v1/analyze/{job_id}/pause")).status_code == 409
                assert (await client.post(f"/api/v1/analyze/{job_id}/resume")).status_code == 409
                assert (await client.post(f"/api/v1/analyze/{job_id}/stop")).status_code == 409
                # 未知 job
                assert (await client.post("/api/v1/analyze/nope/pause")).status_code == 404

        with patchers[0], patchers[1]:
            asyncio.run(scenario())

    def test_stop_interrupts_llm_calls_and_suppresses_events(self):
        """Stop must sink to LLM/tool call boundaries:

        - 停止后不再启动新的 LLM 调用（计数不再增长）；
        - 停止后清空 checkpointer；
        - 停止后新连接的 SSE 只有 started/stopped（工具/LLM 事件已被屏蔽）。
        """
        from tradingagents.graph.checkpointer import has_checkpoint

        patchers, analyst, stop_event = _patched_blocking_fakes()

        async def scenario():
            async with httpx.AsyncClient(transport=TRANSPORT, base_url="http://test") as client:
                resp = await client.post("/api/v1/analyze", json={
                    "ticker": "600519.SH",
                    "date": "2026-08-20",
                    "analysts": ["fundamentals"],
                    "depth": 1,
                    "language": "Chinese",
                })
                job_id = resp.json()["job_id"]
                # 自然完成约 2.4s（每次 LLM 调用 0.6s × 4）；1.2s 时仍在运行中
                await asyncio.sleep(1.2)
                r = await client.post(f"/api/v1/analyze/{job_id}/stop")
                assert r.status_code == 200
                # 模拟真实聊天模型的 on_chat_model_start 边界检查：
                # 停止信号下沉到每次 LLM 调用边界
                stop_event.set()
                res = await _wait_status(client, job_id, {"stopped"}, timeout=30)
                assert res["status"] == "stopped"
                calls_at_stop = analyst.n_calls
                await asyncio.sleep(1.5)
                assert analyst.n_calls == calls_at_stop, (
                    f"LLM 调用在停止后仍在继续: {calls_at_stop} -> {analyst.n_calls}"
                )
                assert not has_checkpoint(
                    os.environ["WEB_CHECKPOINT_DIR"], "600519.SH", "2026-08-20"
                )
                # 停止后新连接：只应收到 started/stopped，无工具/LLM 事件泄漏
                events = await _collect_stream(client, job_id)
                kinds = {e["event"] for e in events}
                assert kinds <= {"started", "stopped"}, kinds

        with patchers[0], patchers[1]:
            asyncio.run(scenario())


@pytest.mark.unit
class TestStopBoundary:
    """停止信号在每次 LLM/工具调用边界抛出 + 停止后事件屏蔽。"""

    def _handler(self, stopped=True):
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        h = api_server.SSEStatsHandler(loop, queue, should_stop=lambda: stopped)
        return h, queue

    def test_raises_at_llm_and_tool_start_boundaries(self):
        h, _ = self._handler()
        with pytest.raises(AnalysisStopped):
            h.on_chat_model_start({"name": "ChatOpenAI"}, "hi")
        with pytest.raises(AnalysisStopped):
            h.on_llm_start({"name": "LLM"}, ["hi"])
        with pytest.raises(AnalysisStopped):
            h.on_tool_start({"name": "tool_game_theory"}, "{}")

    def test_events_suppressed_after_stop(self):
        h, queue = self._handler()
        h._push({"event": "tool", "status": "running", "name": "x", "args": ""})
        assert queue.empty()

    def test_no_raise_when_not_stopped(self):
        h, _ = self._handler(stopped=False)
        h.on_chat_model_start({"name": "ChatOpenAI"}, "hi")
        h.on_tool_start({"name": "tool_game_theory"}, "{}")


@pytest.mark.unit
class TestLLMTokenStream:
    """流式 token 回调：实时推送 + 节点归属 + 停止后屏蔽。"""

    def _handler(self, stopped=False):
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        h = api_server.SSEStatsHandler(loop, queue, should_stop=lambda: stopped)
        return h, loop, queue

    def test_tokens_stream_attributed_to_node(self):
        h, loop, queue = self._handler()
        h.on_chat_model_start({"name": "ChatOpenAI"}, "user: 分析", run_id="run-t1", metadata={
            "langgraph_checkpoint_ns": "Analyst Team:abc",
            "langgraph_node": "Fundamentals Analyst",
        })
        h.on_llm_new_token("你好", run_id="run-t1")
        h.on_llm_new_token("世界", run_id="run-t1")
        loop.run_until_complete(asyncio.sleep(0))  # 驱动 call_soon_threadsafe
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert [e["event"] for e in events] == ["llm_token", "llm_token"]
        assert events[0]["node"] == "Analyst Team/Fundamentals Analyst"
        assert "".join(e["token"] for e in events) == "你好世界"

    def test_tokens_suppressed_after_stop(self):
        h, loop, queue = self._handler(stopped=True)
        h.on_llm_new_token("x", run_id="r1")
        loop.run_until_complete(asyncio.sleep(0))
        assert queue.empty()


@pytest.mark.unit
class TestToolEventAttribution:
    """工具事件带节点归属 / 请求参数 / 结果 / 用途（活动流可解释每次调用）。"""

    def _handler(self):
        loop = asyncio.new_event_loop()
        queue = asyncio.Queue()
        h = api_server.SSEStatsHandler(loop, queue)
        return h, loop, queue

    def test_tool_events_carry_node_args_consumer_and_result(self):
        h, loop, queue = self._handler()
        h.on_tool_start({"name": "tool_technical"},
                        '{"ts_code": "600519.SH", "end_date": "2026-08-23"}',
                        run_id="tool-r1", metadata={
                            "langgraph_checkpoint_ns": "Analyst Team:abc|Technical Analyst:xyz",
                            "langgraph_node": "tools_technical",
                        })
        h.on_tool_end('{"data": {"latest_close_cny": 1291.5}}', run_id="tool-r1")
        loop.run_until_complete(asyncio.sleep(0))
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        running, done = events
        assert running["event"] == "tool" and running["status"] == "running"
        assert running["node"] == "Analyst Team/Technical Analyst/tools_technical"
        assert running["consumer"] == "技术面分析师撰写报告 + 事实校验"
        assert done["status"] == "done"
        assert done["node"] == "Analyst Team/Technical Analyst/tools_technical"
        assert "600519.SH" in done["args"]
        assert done["consumer"] == "技术面分析师撰写报告 + 事实校验"
        assert done["result_chars"] > 0 and "latest_close_cny" in done["result"]

    def test_factchecker_tool_consumer_label(self):
        h, loop, queue = self._handler()
        h.on_tool_start({"name": "tool_game_theory"}, "{}", run_id="tool-r2", metadata={
            "langgraph_checkpoint_ns": "Analyst Team:abc|Game_Theory Analyst:xyz",
            "langgraph_node": "FactChecker-Game_Theory",
        })
        loop.run_until_complete(asyncio.sleep(0))
        ev = queue.get_nowait()
        assert ev["consumer"] == "事实校验（代码核对来源/复算）"
        assert ev["node"] == "Analyst Team/Game_Theory Analyst/FactChecker-Game_Theory"
