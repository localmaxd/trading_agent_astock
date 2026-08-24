"""TradingAgents FastAPI server: async execution + SSE streaming + web UI.

Endpoints:
  GET  /                              → Web UI (web/index.html)
  POST /api/v1/analyze                → Start analysis (ticker + date), return job_id
  GET  /api/v1/analyze/{job_id}/stream → SSE stream: real-time agent progress,
                                          tool / LLM activity (with token usage),
                                          fact-verification results per analyst,
                                          ticker normalization, final decision
  GET  /api/v1/analyze/{job_id}/result → Final result
  POST /api/v1/analyze/{job_id}/pause  → Pause run: progress is persisted to the
                                          per-ticker checkpointer (LangGraph)
  POST /api/v1/analyze/{job_id}/resume → Resume from the checkpointer
  POST /api/v1/analyze/{job_id}/stop   → Stop run: interrupt remaining LLM
                                          execution and clear the checkpointer
  GET  /api/v1/jobs                    → List jobs
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from langgraph.graph.state import CompiledStateGraph

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.errors import AnalysisStopped
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.checkpointer import clear_checkpoint, get_checkpointer, thread_id
from cli.stats_handler import StatsCallbackHandler

app = FastAPI(
    title="TradingAgents API + Web",
    description="Multi-agent LLM financial trading framework with observable fact-checking",
    version="0.2.4",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (replace with Redis in production)
_jobs: Dict[str, Dict[str, Any]] = {}
_job_events: Dict[str, asyncio.Queue] = {}

_executor = ThreadPoolExecutor(max_workers=4)

_ANALYST_NAMES = {
    "fundamentals_report": "Fundamentals Analyst (基本面)",
    "technical_report": "Technical Analyst (技术面)",
    "game_theory_report": "Game Theory Analyst (博弈面)",
    "news_sentiment_report": "News Sentiment Analyst (新闻舆情)",
    "macro_environment_report": "Macro Analyst (宏观环境)",
}

_ANALYST_TYPES = ["fundamentals", "technical", "game_theory", "news_sentiment", "macro"]


# ---------------------------------------------------------------------------
# Graph structure introspection (directed-graph view in the web UI)
# ---------------------------------------------------------------------------

_GRAPH_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _server_config() -> Dict[str, Any]:
    """Base config for server-side graph construction / analysis runs."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = os.environ.get("WEB_LLM_PROVIDER", "deepseek")
    config["quick_think_llm"] = os.environ.get("WEB_QUICK_MODEL", "deepseek-v4-flash")
    config["deep_think_llm"] = os.environ.get("WEB_DEEP_MODEL", "deepseek-v4-flash")
    config["web_search_enabled"] = (
        os.environ.get("WEB_SEARCH_ENABLED", "false").lower() == "true"
    )
    # 暂停/继续使用的 checkpointer 目录；默认随 data_cache_dir，可用
    # WEB_CHECKPOINT_DIR 覆盖（测试/多实例部署时隔离检查点）。
    checkpoint_dir = os.environ.get("WEB_CHECKPOINT_DIR")
    if checkpoint_dir:
        config["data_cache_dir"] = checkpoint_dir
    return config


def _node_kind(name: str) -> str:
    """Classify a graph node for the UI (shape / colour / label)."""
    if name in ("__start__", "__end__"):
        return "pseudo"
    if name.startswith("TickerGuard-"):
        return "guard"
    if name.startswith("tools_"):
        return "tool"
    if name.startswith("FactChecker-"):
        return "factchecker"
    if name.startswith("RetryClear-"):
        return "retry"
    return "agent"


def _serialize_graph(selected_analysts: List[str]) -> Dict[str, Any]:
    """Introspect the compiled LangGraph and serialize nested subgraphs.

    The parent graph is a thin orchestrator whose nodes are the five compiled
    stage subgraphs.  The Analyst Team contains another compiled level: four
    analyst subgraphs, each carrying its own tool / verification / retry
    nodes.  A node with ``kind == 'subgraph'`` therefore includes a recursive
    ``subgraph`` payload for the web UI.

    Results are cached per analyst selection (construction is ~1s).
    """
    key = tuple(sorted(selected_analysts))
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    ta = TradingAgentsGraph(selected_analysts=selected_analysts, config=_server_config())
    g = ta.graph.get_graph()

    def _edge(e):
        return {
            "source": e.source,
            "target": e.target,
            "conditional": bool(getattr(e, "conditional", False)),
        }

    def _compiled_payload(compiled: CompiledStateGraph) -> Dict[str, Any]:
        graph = compiled.get_graph()
        nodes = []
        for node_id, node in graph.nodes.items():
            data = getattr(node, "data", None)
            is_subgraph = isinstance(data, CompiledStateGraph)
            item = {
                "id": node_id,
                "kind": "subgraph" if is_subgraph else _node_kind(node_id),
            }
            if is_subgraph:
                item["subgraph"] = _compiled_payload(data)
            nodes.append(item)
        return {
            "nodes": nodes,
            "edges": [_edge(edge) for edge in graph.edges],
        }

    is_stage = {
        nid: isinstance(getattr(node, "data", None), CompiledStateGraph)
        for nid, node in g.nodes.items()
    }
    parent = {
        "nodes": [
            {"id": nid, "kind": "stage" if is_stage[nid] else _node_kind(nid)}
            for nid in g.nodes
        ],
        "edges": [_edge(e) for e in g.edges],
    }
    stages = []
    for nid, node in g.nodes.items():
        data = getattr(node, "data", None)
        if not isinstance(data, CompiledStateGraph):
            continue
        stages.append({"id": nid, **_compiled_payload(data)})

    result = {"parent": parent, "stages": stages}
    _GRAPH_CACHE[key] = result
    return result


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. 600519.SH")
    date: str = Field(..., description="Analysis date YYYY-MM-DD")
    analysts: List[str] = Field(
        default=["fundamentals", "technical", "game_theory", "news_sentiment", "macro"],
        description="Analyst types to include",
    )
    depth: int = Field(default=1, ge=1, le=5, description="Debate rounds")
    language: str = Field(default="Chinese", description="Output language")
    checkpoint: bool = Field(default=False, description="Enable checkpoint/resume")


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    ticker: str
    date: str
    created_at: str


class JobResult(BaseModel):
    job_id: str
    status: str
    ticker: str
    date: str
    created_at: str
    completed_at: Optional[str] = None
    decision: Optional[str] = None
    reports: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Live observability callback: pushes tool / LLM activity into the SSE queue
# ---------------------------------------------------------------------------


def _push_event(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, event: Dict[str, Any]) -> None:
    """Thread-safe push of an SSE event.

    Prefers call_soon_threadsafe; if the loop is already closed (shutdown or
    a short-lived test client), falls back to a direct put_nowait so the
    analysis thread never dies on teardown.
    """
    try:
        loop.call_soon_threadsafe(queue.put_nowait, event)
    except RuntimeError:
        try:
            queue.put_nowait(event)
        except Exception:
            pass


def _excerpt(text: Any, limit: int = 400) -> str:
    """Single-line excerpt of an LLM input / output for the UI."""
    s = str(text) if text is not None else ""
    s = " ".join(s.split())
    return s[:limit] + ("..." if len(s) > limit else "")


def _node_identity(metadata: Optional[Dict[str, Any]] = None) -> tuple[str, List[str]]:
    """Resolve the exact LangGraph node from callback/stream metadata.

    Token accounting must be bound at LLM callback time. Pairing completed
    calls to later ``messages`` events by FIFO is unsafe because analyst
    branches run concurrently and non-LLM messages (for example RetryClear's
    ``HumanMessage('Continue')``) also pass through the stream.
    """
    metadata = metadata or {}
    checkpoint_ns = metadata.get("langgraph_checkpoint_ns") or ""
    path = [
        part.split(":", 1)[0]
        for part in checkpoint_ns.split("|")
        if part
    ]
    leaf = metadata.get("langgraph_node") or ""
    if leaf and (not path or path[-1] != leaf):
        path.append(leaf)
    return "/".join(path), path


def _tool_consumer(name: str, node: str) -> str:
    """工具返回的数据被谁使用（用于前端活动流展示）。"""
    if "FactChecker" in (node or ""):
        return "事实校验（代码核对来源/复算）"
    consumers = {
        "tool_fundamental": "基本面分析师撰写报告 + 事实校验",
        "tool_technical": "技术面分析师撰写报告 + 事实校验",
        "tool_game_theory": "博弈面分析师撰写报告 + 事实校验",
        "tool_news_sentiment": "新闻舆情分析师撰写报告",
        "tool_market_environment": "宏观分析师撰写报告",
        "tool_schema": "分析师理解字段含义（调用数据接口前）",
        "position": "Trader 交易决策（持仓/资金判断）",
    }
    return consumers.get(name, "")


class SSEStatsHandler(StatsCallbackHandler):
    """Stats handler that records every LLM call (input/output/tokens/duration)
    keyed by its LangChain run_id, streams tool activity, and pushes stats.

    The graph streams with stream_mode=["updates", "messages"]; the messages
    events carry (message, metadata) with langgraph_node and the message's
    run_id, so the streaming loop merges the recorded LLM details with the
    node name and emits one llm_call event per LLM invocation.

    Callbacks fire inside the executor thread; events are marshalled to the
    asyncio loop with call_soon_threadsafe.

    Run control: ``should_stop`` (optional callable) is checked at every LLM
    and tool START boundary.  When it returns True the handler raises
    :class:`tradingagents.errors.AnalysisStopped`, which aborts the whole
    graph stream — the stop signal therefore sinks into every LLM / tool /
    fact-checker call, including inside parallel analyst subgraphs.  All
    event pushes are suppressed once stopped, so no late activity leaks to
    the frontend after the "stopped" terminal event.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        should_stop=None,
    ):
        super().__init__()
        self._loop = loop
        self._queue = queue
        self._should_stop = should_stop or (lambda: False)
        # run_id -> record; on_llm_end fills output/tokens/duration.
        # _done keeps the COMPLETION ORDER of all records (as run_id refs or
        # the record objects for run_id-less LLMs) so the streaming loop can
        # pair streamed messages (AIMessageChunks have no run_id) with the
        # right record FIFO-style.
        self._llm_records: Dict[str, Dict[str, Any]] = {}
        self._done: list = []
        # run_id -> 工具启动时的节点归属（on_tool_end 无 metadata，按 run_id 回查）
        self._tool_records: Dict[str, Dict[str, Any]] = {}
        self._current_tool_rec: Optional[Dict[str, Any]] = None

    def _check_stop(self) -> None:
        """Abort at the call boundary when the user stopped the run."""
        if self._should_stop():
            raise AnalysisStopped("analysis stopped by user")

    def _push(self, event: Dict[str, Any]) -> None:
        # 停止后屏蔽所有回调事件，避免前端在 stopped 之后继续收到活动
        if self._should_stop():
            return
        _push_event(self._loop, self._queue, event)

    def _record_input(self, run_id, messages, metadata=None) -> Dict[str, Any]:
        excerpt = _excerpt(messages, 400) if messages else ""
        node, path = _node_identity(metadata)
        rec = {"input": excerpt, "ts": time.time(), "output": "", "tokens": None,
               "duration": None, "node": node, "path": path}
        if run_id is not None:
            self._llm_records[str(run_id)] = rec
        else:
            self._done.append(("rec", rec))
        # run_id-less providers 的 token 回调按"最近一次开始的调用"归属
        self._current_rec = rec
        return rec

    def _records_by_item(self, item) -> Optional[Dict[str, Any]]:
        kind, key = item
        if kind == "id":
            return self._llm_records.get(key)
        return key

    def _finish_record(self, run_id, output, tokens_in, tokens_out) -> Optional[Dict[str, Any]]:
        """Fill output/tokens/duration and append the record to the done queue."""
        rec = None
        if run_id is not None and str(run_id) in self._llm_records:
            rec = self._llm_records[str(run_id)]
        elif self._done:
            rec = self._records_by_item(self._done.pop(0))
        if rec is None:
            rec = {"input": "", "ts": time.time(), "output": "", "tokens": None,
                   "duration": None}
            if run_id is not None:
                self._llm_records[str(run_id)] = rec
        rec["output"] = _excerpt(output, 600)
        rec["tokens"] = {"in": tokens_in, "out": tokens_out}
        rec["duration"] = round(time.time() - rec["ts"], 2)
        if run_id is not None and str(run_id) in self._llm_records:
            self._done.append(("id", str(run_id)))
        else:
            self._done.append(("rec", rec))
        return rec

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._check_stop()
        super().on_chat_model_start(serialized, messages, **kwargs)
        self._record_input(
            kwargs.get("run_id"), messages, kwargs.get("metadata")
        )

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._check_stop()
        super().on_llm_start(serialized, prompts, **kwargs)
        self._record_input(
            kwargs.get("run_id"), prompts, kwargs.get("metadata")
        )

    def on_llm_end(self, response, **kwargs):
        tokens_in = tokens_out = 0
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            generation = None
        if generation is not None and hasattr(generation, "message"):
            usage = getattr(generation.message, "usage_metadata", None) or {}
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
        super().on_llm_end(response, **kwargs)
        # The record stays queued until the streaming loop sees the message
        # (with its langgraph_node) and emits the llm_call event.
        self._finish_record(
            kwargs.get("run_id"),
            generation.message.content if generation is not None and hasattr(generation, "message") else "",
            tokens_in, tokens_out,
        )

    def on_llm_new_token(self, token, **kwargs):
        """流式 token 回调：实时推送模型输出片段（带节点归属）。

        LLM 客户端已开启 streaming=True，模型每吐出一个 token 就推送一个
        ``llm_token`` 事件，前端活动流可以实时显示生成内容，而不是等整个
        LLM 调用结束才出现。停止后不再推送。
        """
        super().on_llm_new_token(token, **kwargs)
        if not token:
            return
        if self._should_stop():
            return
        rec = None
        run_id = kwargs.get("run_id")
        if run_id is not None:
            rec = self._llm_records.get(str(run_id))
        if rec is None:
            # run_id-less providers：归属到最近一次开始的调用
            rec = getattr(self, "_current_rec", None)
        node = (rec or {}).get("node") or "llm"
        self._push({
            "event": "llm_token",
            "node": node,
            "token": token,
            "run_id": str(run_id) if run_id is not None else "",
        })

    def consume_record(self, run_id=None) -> Optional[Dict[str, Any]]:
        """Pop the next LLM record: exact run_id match first, then FIFO.

        Streamed messages are often AIMessageChunks WITHOUT a run_id, so the
        FIFO path (completion order) is what pairs real LLM calls with their
        messages. One record is consumed by the first chunk of each call;
        later chunks find nothing and are ignored.
        """
        if run_id is not None and str(run_id) in self._llm_records:
            return self._llm_records.pop(str(run_id))
        while self._done:
            item = self._done.pop(0)
            rec = self._records_by_item(item)
            if rec is not None:
                if item[0] == "id":
                    self._llm_records.pop(item[1], None)
                return rec
        return None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._check_stop()
        super().on_tool_start(serialized, input_str, **kwargs)
        name = serialized.get("name", "tool")
        metadata = kwargs.get("metadata") or {}
        node, path = _node_identity(metadata)
        rec = {"name": name, "node": node or "unknown", "path": path,
               "args": input_str, "consumer": _tool_consumer(name, node or ""),
               "ts": time.time()}
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._tool_records[str(run_id)] = rec
        self._current_tool_rec = rec
        self._push({
            "event": "tool", "status": "running",
            "name": name,
            "node": node or "unknown",
            "path": path,
            "args": input_str[:200],
            "consumer": rec["consumer"],
        })

    def _tool_record(self, kwargs) -> Dict[str, Any]:
        """工具结束回调没有 metadata：按 run_id 回查启动时的节点归属。"""
        run_id = kwargs.get("run_id")
        if run_id is not None:
            rec = self._tool_records.get(str(run_id))
            if rec is not None:
                return rec
        return getattr(self, "_current_tool_rec", None) or {
            "name": "tool", "node": "unknown", "path": [], "args": "",
            "consumer": "", "ts": time.time(),
        }

    def on_tool_end(self, output, **kwargs):
        super().on_tool_end(output, **kwargs)
        # 并行分析师分支并发执行：get_activity()["tools"][-1] 是全局列表，
        # 不能用于归属——一律用 run_id 记录的启动信息（name/node/args/用途）。
        rec = self._tool_record(kwargs)
        out_str = (
            output
            if isinstance(output, str)
            else getattr(output, "content", output)
        )
        chars = len(str(out_str)) if out_str is not None else 0
        elapsed = round(time.time() - rec.get("ts", time.time()), 2)
        self._push({
            "event": "tool", "status": "done",
            "name": rec["name"],
            "node": rec["node"],
            "path": rec["path"],
            "args": rec["args"][:200],
            "consumer": rec["consumer"],
            "detail": f"({elapsed}s, {chars} ch)",
            "result": _excerpt(out_str, 300),
            "result_chars": chars,
            "stats": self.get_stats(),
        })

    def on_tool_error(self, error, **kwargs):
        super().on_tool_error(error, **kwargs)
        rec = self._tool_record(kwargs)
        self._push({"event": "tool", "status": "error", "name": rec["name"],
                    "node": rec["node"], "path": rec["path"], "args": rec["args"][:200],
                    "consumer": rec["consumer"],
                    "detail": str(error)[:120]})


# ---------------------------------------------------------------------------
# Chunk -> SSE event extraction
# ---------------------------------------------------------------------------


def _extract_update_events(
    update: Dict[str, Any],
    merged: Dict[str, Any],
    seen_reports: Dict[str, str],
    seen_verification: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Extract observable events from an updates-mode payload {node: update}.

    update is a PARTIAL state update (only the channels the node wrote), so
    diffing uses the accumulated merged state and per-key seen markers.

    Node start/completion lifecycle (which drives the graph animation) is
    emitted separately from the "debug" stream mode, so this function only
    extracts domain events: ticker, reports, claims, verification, decisions.
    """
    events: List[Dict[str, Any]] = []

    # 1) Ticker normalization by the TickerGuard nodes
    ticker = update.get("company_of_interest")
    if ticker and merged.get("company_of_interest") != ticker:
        events.append({
            "event": "ticker",
            "ticker": ticker,
            "previous": merged.get("company_of_interest", ""),
            "note": "normalized by TickerGuard",
        })

    # 2) Analyst reports completed
    for key, name in _ANALYST_NAMES.items():
        val = update.get(key)
        if val and val.strip() and seen_reports.get(key) != val:
            events.append({
                "event": "agent_completed",
                "agent": name,
                "report_key": key,
                "report": val,
            })

    # 3) Structured claims produced by an analyst
    for analyst_type, claim_key in {
        "fundamentals": "fundamentals_claims",
        "technical": "technical_claims",
        "game_theory": "game_theory_claims",
    }.items():
        claims = update.get(claim_key) or []
        if claims:
            events.append({"event": "claims", "analyst": analyst_type, "claims": claims})

    # 4) Fact-verification results (each partial update carries the analyst key)
    vs = update.get("verification_state") or {}
    for analyst_type, entry in vs.items():
        if entry and seen_verification.get(analyst_type, -1) != entry.get("attempts"):
            events.append({
                "event": "verification",
                "analyst": analyst_type,
                "attempts": entry.get("attempts"),
                "passed": entry.get("passed"),
                "feedback": entry.get("feedback", ""),
                "items": entry.get("items", []),
                "report_md": entry.get("report_md", ""),
            })

    # 5) Research debate / Trader / PM decisions
    #    每个阶段子图输出时都会携带共享通道（如 trader_investment_plan 被
    #    Trader/Risk Debate/PM 三个阶段的 schema 声明），所以必须按内容去重，
    #    否则同一份 Trader 报告会在后续阶段更新里重复推送多次。
    debate = update.get("investment_debate_state") or {}
    if debate.get("judge_decision") and debate["judge_decision"].strip():
        judge = debate["judge_decision"]
        if seen_reports.get("research_judge") != judge:
            seen_reports["research_judge"] = judge
            events.append({
                "event": "agent_completed",
                "agent": "Research Manager",
                "report_key": "investment_plan",
                "report": judge,
                # 辩论全文一并推送（前端报告区展示复盘论据）
                "debate_bull": debate.get("bull_history", ""),
                "debate_bear": debate.get("bear_history", ""),
            })

    plan = update.get("trader_investment_plan") or ""
    if plan and plan.strip():
        if seen_reports.get("trader_investment_plan") != plan:
            seen_reports["trader_investment_plan"] = plan
            events.append({
                "event": "agent_completed",
                "agent": "Trader",
                "report_key": "trader_investment_plan",
                "report": plan,
            })

    risk = update.get("risk_debate_state") or {}
    if risk.get("judge_decision") and risk["judge_decision"].strip():
        judge = risk["judge_decision"]
        if seen_reports.get("pm_judge") != judge:
            seen_reports["pm_judge"] = judge
            events.append({
                "event": "decision",
                "agent": "Portfolio Manager",
                "report": judge,
                # 三方风控辩论全文一并推送（前端决策区展示）
                "risk_aggressive": risk.get("aggressive_history", ""),
                "risk_conservative": risk.get("conservative_history", ""),
                "risk_neutral": risk.get("neutral_history", ""),
            })

    return events


def _clear_job_checkpoint(job: Dict[str, Any]) -> None:
    """Close the job's checkpointer context and delete its checkpoint rows.

    暂停/停止/完成 后调用：保证「继续」总是从 checkpointer 恢复，而「停止」
    一定清空 checkpointer。
    """
    ctx = job.get("checkpointer_ctx")
    if ctx is not None:
        job["checkpointer_ctx"] = None
        try:
            ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - 清理失败不阻塞
            pass
    cache_dir = job.get("cache_dir")
    ticker = job.get("ticker")
    job_date = job.get("date")
    if cache_dir and ticker and job_date:
        try:
            clear_checkpoint(cache_dir, ticker, str(job_date))
        except Exception:  # noqa: BLE001
            pass


def _stop_job(job_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """Terminate a job: clear the checkpoint, mark it stopped, push the event."""
    job = _jobs[job_id]
    _clear_job_checkpoint(job)
    job["graph"] = None
    job["status"] = "stopped"
    _push_event(loop, _job_events[job_id], {"event": "stopped", "reason": "user stopped"})


def _finish_job_from_state(job: Dict[str, Any], final_state: Dict[str, Any]) -> None:
    """Mark a job completed and build its final result payload."""
    job["status"] = "completed"
    job["completed_at"] = datetime.now().isoformat()
    job["result"] = {
        "decision": _extract_decision(final_state),
        "reports": _build_reports(final_state),
        "verification": final_state.get("verification_state", {}),
        "full_state": _serialize_state(final_state),
    }


def _run_analysis_sync(
    loop: asyncio.AbstractEventLoop,
    job_id: str,
    ticker: str,
    date: str,
    config: Dict[str, Any],
    resume: bool = False,
) -> Dict[str, Any]:
    """Stream TradingAgentsGraph in a thread, pushing observable events.

    Mirrors the CLI's live loop: the graph streams chunk by chunk and every
    chunk is diffed into SSE events (progress, claims, verification, ...).
    The SSEStatsHandler additionally pushes tool/LLM activity as it happens.

    Three stream modes are combined with subgraphs=True so the web UI gets:

      - "updates": per-node partial state writes. At the parent level each
        stage subgraph yields its full merged output (reports, claims,
        verification, decisions); subgraph-level payloads are ignored here
        (same events arrive with the parent-level output, deduplicated by
        the seen markers).
      - "messages": every LLM message with (langgraph_node,
        langgraph_checkpoint_ns) so each LLM call is attributed to the exact
        node -- including nested analyst internals, e.g. "Analyst Team /
        Fundamentals Analyst / Analyze".
      - "debug": precise task start/stop events for EVERY node at every
        level, driving the directed-graph flow animation (node_started /
        node) and per-node wall-time stats.

    Run control (pause/resume/stop) via the job's ``control`` dict:

      - pause: the loop breaks at the next chunk boundary; the per-ticker
        SqliteSaver keeps the latest super-step, and the job stays
        resumable (status=``paused``).
      - resume: ``_run_analysis_sync(..., resume=True)`` re-streams the same
        compiled graph with the same thread_id, so LangGraph continues from
        the checkpoint.
      - stop: the handler raises AnalysisStopped at the NEXT LLM / tool call
        boundary (inside parallel analyst subgraphs and fact-checkers too),
        aborting the whole stream; in-flight calls run to completion but no
        new call starts.  The checkpoint is CLEARED and the job is marked
        ``stopped``; events pushed after the stop are suppressed.
    """
    job = _jobs[job_id]
    control = job.get("control", {})
    queue = _job_events[job_id]
    # 复用同一 handler：第一次运行绑定了 LLM 回调，resume 时继续用它记账。
    # should_stop 闭包指向 job 的 control，停止信号因此下沉到每次
    # LLM / 工具 / 事实校验调用的边界。
    handler = job.get("handler")
    if handler is None:
        handler = SSEStatsHandler(
            loop, queue, should_stop=lambda: bool(control.get("stop"))
        )
        job["handler"] = handler

    ta = job.get("graph")
    if ta is None:
        ta = TradingAgentsGraph(
            selected_analysts=config.get(
                "selected_analysts",
                ["fundamentals", "technical", "game_theory", "news_sentiment", "macro"],
            ),
            config=config,
            callbacks=[handler],
        )
        job["graph"] = ta
        job["cache_dir"] = config["data_cache_dir"]
        job["thread_id"] = thread_id(ticker, str(date))
        # 全新任务：清掉可能残留的同 ticker+date 检查点，从干净状态开始
        try:
            clear_checkpoint(config["data_cache_dir"], ticker, str(date))
        except Exception:  # noqa: BLE001
            pass
        # 暂停/继续依赖 per-ticker SqliteSaver 检查点
        ctx = get_checkpointer(config["data_cache_dir"], ticker)
        job["checkpointer_ctx"] = ctx
        saver = ctx.__enter__()
        ta.graph = ta.workflow.compile(checkpointer=saver)

    init_state = ta.propagator.create_initial_state(ticker, date)
    args = ta.propagator.get_graph_args(callbacks=[handler])
    args["stream_mode"] = ["updates", "messages", "debug"]
    args["subgraphs"] = True
    # 同一 thread_id：resume 时 LangGraph 自动从 checkpointer 恢复
    args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = job["thread_id"]

    if resume:
        _push_event(loop, queue, {"event": "resumed", "job_id": job_id})
    else:
        _push_event(loop, queue, {"event": "analysis_started", "ticker": ticker, "date": date})

    merged: Dict[str, Any] = {}
    seen_reports: Dict[str, str] = {}
    seen_verification: Dict[str, int] = {}
    node_stats: Dict[str, Dict[str, Any]] = {}
    node_started_ts: Dict[str, float] = {}
    last_event_ts: float = time.time()

    def _add_node_token(node: str, tokens: Dict[str, int], duration: Optional[float]) -> None:
        stats = node_stats.setdefault(node, {
            "calls": 0, "tokens_in": 0, "tokens_out": 0,
            "time_ms": 0.0, "llm_time_ms": 0.0,
        })
        stats["calls"] += 1
        stats["tokens_in"] += tokens.get("in", 0) if tokens else 0
        stats["tokens_out"] += tokens.get("out", 0) if tokens else 0
        if duration is not None:
            stats["llm_time_ms"] += duration * 1000

    def _ns_path(ns) -> List[str]:
        """Namespace tuple -> plain node path.

        With subgraphs=True LangGraph prefixes every level with a unique id
        ("Analyst Team:<uuid>"); the part before ':' is the real node name.
        """
        return [part.split(":", 1)[0] for part in (ns or ())]

    for item in ta.graph.stream(init_state, **args):
        if control.get("stop"):
            # 停止：中断所有后续执行，清空 checkpointer
            _stop_job(job_id, loop)
            return {}
        if control.get("pause"):
            # 暂停：当前进度（最近一个 super-step）已写入 checkpointer，
            # 继续时将从此恢复
            job["status"] = "paused"
            _push_event(loop, queue, {"event": "paused"})
            return {}

        ns, mode, payload = item

        if mode == "debug":
            # task -> node started; task_result -> node completed.
            # payload: {"type": "task"|"task_result", "payload": {"name": ...}}
            ev_type = payload.get("type")
            name = (payload.get("payload") or {}).get("name")
            if not name:
                continue
            path = _ns_path(ns) + [name]
            full = "/".join(path)
            if ev_type == "task":
                node_started_ts[full] = time.time()
                _push_event(loop, queue, {
                    "event": "node_started",
                    "node": full,
                    "path": path,
                })
            elif ev_type == "task_result":
                started = node_started_ts.pop(full, None)
                started = started if started is not None else last_event_ts
                duration = round(time.time() - started, 2)
                stats = node_stats.setdefault(full, {
                    "calls": 0, "tokens_in": 0, "tokens_out": 0,
                    "time_ms": 0.0, "llm_time_ms": 0.0,
                })
                stats["time_ms"] += duration * 1000
                last_event_ts = time.time()
                _push_event(loop, queue, {
                    "event": "node",
                    "node": full,
                    "path": path,
                    "status": "completed",
                    "duration": duration,
                    "node_stats": {k: dict(v) for k, v in node_stats.items()},
                })

        elif mode == "updates":
            if ns:
                # 子图内部 partial writes：只抽取 FactChecker 的 verification 事件
                # 实时推送（父级输出到达时按 attempts 去重，不会重复），
                # 这样即使阶段未完成/被暂停/停止，校验结论也已可见。
                for node, update in (payload or {}).items():
                    update = update or {}
                    vs = update.get("verification_state") or {}
                    for analyst_type, entry in vs.items():
                        if not entry:
                            continue
                        if seen_verification.get(analyst_type, -1) == entry.get("attempts"):
                            continue
                        seen_verification[analyst_type] = entry.get("attempts")
                        _push_event(loop, queue, {
                            "event": "verification",
                            "analyst": analyst_type,
                            "attempts": entry.get("attempts"),
                            "passed": entry.get("passed"),
                            "feedback": entry.get("feedback", ""),
                            "items": entry.get("items", []),
                            "report_md": entry.get("report_md", ""),
                        })
                continue
            for node, update in (payload or {}).items():
                update = update or {}  # nodes that wrote nothing emit None
                for event in _extract_update_events(update, merged, seen_reports, seen_verification):
                    _push_event(loop, queue, event)
                merged.update(update)
                for key in _ANALYST_NAMES:
                    if update.get(key):
                        seen_reports[key] = update[key]
                vs = update.get("verification_state") or {}
                for analyst_type, entry in vs.items():
                    if entry:
                        seen_verification[analyst_type] = entry.get("attempts")

        else:  # messages
            msg, meta = payload
            stream_node, stream_path = _node_identity(meta)
            run_id = getattr(msg, "run_id", None)
            rec = handler.consume_record(run_id)
            if rec is not None:
                # The callback record owns the node identity. The message
                # metadata is only a fallback for providers that omit callback
                # metadata; this prevents RetryClear/non-LLM messages from
                # stealing the next parallel analyst's token record.
                node = rec.get("node") or stream_node or "unknown"
                path = rec.get("path") or stream_path or [node]
                tokens = rec.get("tokens") or {}
                output = rec.get("output") or _excerpt(getattr(msg, "content", ""), 600)
                duration = rec.get("duration")
                _add_node_token(node, tokens, duration)
                _push_event(loop, queue, {
                    "event": "llm_call",
                    "node": node,
                    "path": path,
                    "input": rec.get("input", ""),
                    "output": output,
                    "tokens": tokens,
                    "duration": duration,
                    "node_stats": {
                        k: dict(v) for k, v in node_stats.items()
                    },
                })
            _push_event(loop, queue, {"event": "stats", "stats": handler.get_stats()})

    # resume 场景下 merged 只包含本次增量，最终状态优先从 checkpointer 读取
    final_state = merged
    try:
        saved = ta.graph.get_state({"configurable": {"thread_id": job["thread_id"]}})
        if saved is not None and saved.values:
            final_state = saved.values
    except Exception:  # noqa: BLE001 - 回退到累计状态
        pass

    # 先落库状态/结果，再推送完成事件（避免 SSE 与 result 端点竞态）
    _finish_job_from_state(job, final_state)
    _push_event(loop, queue, {
        "event": "completed",
        "decision": _extract_decision(final_state),
        "final_trade_decision": final_state.get("final_trade_decision", ""),
        "stats": handler.get_stats(),
        "node_stats": {k: dict(v) for k, v in node_stats.items()},
    })
    # 运行完成：清空检查点并释放图实例
    _clear_job_checkpoint(job)
    job["graph"] = None
    return final_state


def _build_reports(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Build structured reports from final state."""
    reports = {}
    for key, name in _ANALYST_NAMES.items():
        if final_state.get(key):
            reports[name] = final_state[key]
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        reports["Research Team"] = {
            "bull": debate.get("bull_history", ""),
            "bear": debate.get("bear_history", ""),
            "manager_decision": debate.get("judge_decision", ""),
        }
    if final_state.get("trader_investment_plan"):
        reports["Trader"] = final_state["trader_investment_plan"]
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        reports["Risk Management"] = {
            "aggressive": risk.get("aggressive_history", ""),
            "conservative": risk.get("conservative_history", ""),
            "neutral": risk.get("neutral_history", ""),
        }
        reports["Portfolio Manager"] = risk.get("judge_decision", "")
    return reports


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def web_ui():
    """Serve the web UI."""
    page = Path(__file__).parent / "web" / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(str(page))


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """Start a new analysis job. Returns immediately with job_id."""
    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    config = _server_config()
    config["max_debate_rounds"] = req.depth
    config["max_risk_discuss_rounds"] = req.depth
    config["output_language"] = req.language
    config["checkpoint_enabled"] = req.checkpoint
    config["selected_analysts"] = req.analysts

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "ticker": req.ticker,
        "date": req.date,
        "created_at": now,
        "completed_at": None,
        "result": None,
        "error": None,
        "config": config,
        # 运行控制：暂停/继续/停止（由分析线程在 chunk 边界读取）
        "control": {"pause": False, "stop": False},
        # 暂停/继续依赖 per-ticker checkpointer
        "graph": None,
        "handler": None,
        "thread_id": None,
        "cache_dir": None,
        "checkpointer_ctx": None,
    }
    _job_events[job_id] = asyncio.Queue()

    # Run the (synchronous) analysis in a dedicated thread. The SSE events are
    # marshalled into the asyncio queue via call_soon_threadsafe, so this works
    # both under uvicorn and under sync test clients (an asyncio.create_task
    # would never be scheduled in the latter).
    loop = asyncio.get_running_loop()

    def _run_sync():
        if _jobs[job_id]["status"] != "pending":
            return
        _jobs[job_id]["status"] = "running"
        try:
            # 完成/暂停/停止/失败状态均由 _run_analysis_sync 内部设置
            _run_analysis_sync(loop, job_id, req.ticker, req.date, config)
        except AnalysisStopped:
            # 停止信号在 LLM/工具调用边界抛出：终止运行并清空检查点
            _stop_job(job_id, loop)
        except Exception as exc:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            _clear_job_checkpoint(job)
            job["graph"] = None
            _push_event(loop, _job_events[job_id], {"event": "error", "error": str(exc)})

    threading.Thread(target=_run_sync, daemon=True).start()

    return AnalyzeResponse(
        job_id=job_id,
        status="pending",
        ticker=req.ticker,
        date=req.date,
        created_at=now,
    )


@app.get("/api/v1/analyze/{job_id}/stream")
async def stream_progress(job_id: str):
    """SSE stream of real-time analysis progress."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = _job_events[job_id]

    async def event_generator():
        yield f"data: {json.dumps({'event': 'started', 'job_id': job_id}, ensure_ascii=False)}\n\n"
        # 暂停/停止后客户端重连：立即告知当前状态并结束（无运行线程再产生事件）
        status = _jobs[job_id]["status"]
        if status == "paused":
            yield f"data: {json.dumps({'event': 'paused'}, ensure_ascii=False)}\n\n"
            return
        if status == "stopped":
            yield f"data: {json.dumps({'event': 'stopped', 'reason': 'user stopped'}, ensure_ascii=False)}\n\n"
            return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=600.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("event") in ("completed", "error", "paused", "stopped"):
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'event': 'timeout'}, ensure_ascii=False)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/analyze/{job_id}/pause")
async def pause_analysis(job_id: str):
    """暂停运行中的分析：当前进度写入 checkpointer（LangGraph super-step）。

    分析线程在下一个 chunk 边界停止；job 状态变为 ``paused``，
    之后可通过 /resume 从 checkpointer 继续。
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("running", "pending"):
        raise HTTPException(status_code=409, detail=f"Cannot pause job in status {job['status']}")
    job["control"]["pause"] = True
    return {"job_id": job_id, "status": "pausing"}


@app.post("/api/v1/analyze/{job_id}/resume")
async def resume_analysis(job_id: str):
    """从 checkpointer 继续已暂停的分析（同一 thread_id 重新 stream）。"""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "paused":
        raise HTTPException(status_code=409, detail=f"Cannot resume job in status {job['status']}")
    job["control"]["pause"] = False
    job["status"] = "running"
    loop = asyncio.get_running_loop()
    config = job["config"]

    def _resume_sync():
        job = _jobs[job_id]
        try:
            # 完成/暂停/停止/失败状态均由 _run_analysis_sync 内部设置
            _run_analysis_sync(
                loop, job_id, job["ticker"], job["date"], config, resume=True
            )
        except AnalysisStopped:
            # 停止信号在 LLM/工具调用边界抛出：终止运行并清空检查点
            _stop_job(job_id, loop)
        except Exception as exc:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            _clear_job_checkpoint(job)
            job["graph"] = None
            _push_event(loop, _job_events[job_id], {"event": "error", "error": str(exc)})

    threading.Thread(target=_resume_sync, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.post("/api/v1/analyze/{job_id}/stop")
async def stop_analysis(job_id: str):
    """停止分析：中断所有后续 LLM 执行，并清空 checkpointer。

    运行中：置 stop 标记，分析线程在下一个 chunk 边界处理；
    已暂停：无运行线程，直接清理 checkpointer 并标记 stopped。
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("running", "paused", "pending"):
        raise HTTPException(status_code=409, detail=f"Cannot stop job in status {job['status']}")
    job["control"]["stop"] = True
    if job["status"] in ("pending", "paused"):
        # 分析线程尚未开始 / 已暂停：直接终止并清空 checkpointer
        _stop_job(job_id, asyncio.get_running_loop())
    return {"job_id": job_id, "status": "stopping"}


@app.get("/api/v1/analyze/{job_id}/result", response_model=JobResult)
async def get_result(job_id: str):
    """Get final analysis result."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    result = job.get("result") or {}

    return JobResult(
        job_id=job_id,
        status=job["status"],
        ticker=job["ticker"],
        date=job["date"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        decision=result.get("decision"),
        reports=result.get("reports"),
        verification=result.get("verification"),
        error=job.get("error"),
    )


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all jobs."""
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "ticker": j["ticker"],
            "date": j["date"],
            "created_at": j["created_at"],
        }
        for j in _jobs.values()
    ]


@app.get("/api/v1/graph")
async def get_graph_structure(
    analysts: str = "fundamentals,technical,game_theory,news_sentiment,macro",
):
    """Directed-graph structure of the compiled workflow for the web UI.

    Returns the parent chain (TickerGuards + stage subgraphs) plus every
    stage's internal nodes and edges, so the browser can draw the full
    pipeline and animate data flow while an analysis runs.
    """
    try:
        selected = [a.strip() for a in analysts.split(",") if a.strip()]
        if not selected:
            raise HTTPException(status_code=400, detail="analysts is empty")
        return _serialize_graph(selected)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _extract_decision(final_state: Dict[str, Any]) -> Optional[str]:
    """Extract final decision string from state."""
    from tradingagents.agents.utils.rating import parse_rating
    decision_text = final_state.get("final_trade_decision", "")
    if decision_text:
        return parse_rating(decision_text)
    return None


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize final state for JSON response (remove non-serializable objects)."""
    result = {}
    for key, value in state.items():
        if key == "messages" or key.endswith("_messages") or key.startswith("messages_"):
            continue
        if isinstance(value, (str, int, float, bool, list, dict)):
            result[key] = value
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
