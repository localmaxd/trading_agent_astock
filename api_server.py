"""TradingAgents FastAPI server: async execution + SSE streaming + web UI.

Endpoints:
  GET  /                              → Web UI (web/index.html)
  POST /api/v1/analyze                → Start analysis (ticker + date), return job_id
  GET  /api/v1/analyze/{job_id}/stream → SSE stream: real-time agent progress,
                                          tool / LLM activity (with token usage),
                                          fact-verification results per analyst,
                                          ticker normalization, final decision
  GET  /api/v1/analyze/{job_id}/result → Final result
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

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
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
}

_ANALYST_TYPES = ["fundamentals", "technical", "game_theory", "news_sentiment"]


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. 600519.SH")
    date: str = Field(..., description="Analysis date YYYY-MM-DD")
    analysts: List[str] = Field(
        default=["fundamentals", "technical", "game_theory", "news_sentiment"],
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


class SSEStatsHandler(StatsCallbackHandler):
    """Stats handler that records every LLM call (input/output/tokens/duration)
    keyed by its LangChain run_id, streams tool activity, and pushes stats.

    The graph streams with stream_mode=["updates", "messages"]; the messages
    events carry (message, metadata) with langgraph_node and the message's
    run_id, so the streaming loop merges the recorded LLM details with the
    node name and emits one llm_call event per LLM invocation.

    Callbacks fire inside the executor thread; events are marshalled to the
    asyncio loop with call_soon_threadsafe.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        super().__init__()
        self._loop = loop
        self._queue = queue
        # run_id -> record; on_llm_end fills output/tokens/duration.
        # _done keeps the COMPLETION ORDER of all records (as run_id refs or
        # the record objects for run_id-less LLMs) so the streaming loop can
        # pair streamed messages (AIMessageChunks have no run_id) with the
        # right record FIFO-style.
        self._llm_records: Dict[str, Dict[str, Any]] = {}
        self._done: list = []

    def _push(self, event: Dict[str, Any]) -> None:
        _push_event(self._loop, self._queue, event)

    def _record_input(self, run_id, messages) -> Dict[str, Any]:
        excerpt = _excerpt(messages, 400) if messages else ""
        rec = {"input": excerpt, "ts": time.time(), "output": "", "tokens": None,
               "duration": None}
        if run_id is not None:
            self._llm_records[str(run_id)] = rec
        else:
            self._done.append(("rec", rec))
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
        super().on_chat_model_start(serialized, messages, **kwargs)
        self._record_input(kwargs.get("run_id"), messages)

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
        super().on_tool_start(serialized, input_str, **kwargs)
        self._push({
            "event": "tool", "status": "running",
            "name": serialized.get("name", "tool"),
            "args": input_str[:120],
        })

    def on_tool_end(self, output, **kwargs):
        super().on_tool_end(output, **kwargs)
        entry = self.get_activity()["tools"][-1]
        self._push({
            "event": "tool", "status": "done", "name": entry["name"],
            "detail": entry["detail"], "stats": self.get_stats(),
        })

    def on_tool_error(self, error, **kwargs):
        super().on_tool_error(error, **kwargs)
        self._push({"event": "tool", "status": "error", "name": "tool",
                    "detail": str(error)[:120]})


# ---------------------------------------------------------------------------
# Chunk -> SSE event extraction
# ---------------------------------------------------------------------------


def _extract_update_events(
    node: str,
    update: Dict[str, Any],
    merged: Dict[str, Any],
    seen_reports: Dict[str, str],
    seen_verification: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Extract observable events from an updates-mode payload {node: update}.

    update is a PARTIAL state update (only the channels the node wrote), so
    diffing uses the accumulated merged state and per-key seen markers.
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
    debate = update.get("investment_debate_state") or {}
    if debate.get("judge_decision") and debate["judge_decision"].strip():
        events.append({
            "event": "agent_completed",
            "agent": "Research Manager",
            "report_key": "investment_plan",
            "report": debate["judge_decision"],
        })

    if update.get("trader_investment_plan") and update["trader_investment_plan"].strip():
        events.append({
            "event": "agent_completed",
            "agent": "Trader",
            "report_key": "trader_investment_plan",
            "report": update["trader_investment_plan"],
        })

    risk = update.get("risk_debate_state") or {}
    if risk.get("judge_decision") and risk["judge_decision"].strip():
        events.append({
            "event": "decision",
            "agent": "Portfolio Manager",
            "report": risk["judge_decision"],
        })

    # 6) Node completion (timing for the per-node stats)
    events.append({"event": "node", "node": node, "status": "completed"})

    return events


def _run_analysis_sync(
    loop: asyncio.AbstractEventLoop,
    job_id: str,
    ticker: str,
    date: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Stream TradingAgentsGraph in a thread, pushing observable events.

    Mirrors the CLI's live loop: the graph streams chunk by chunk and every
    chunk is diffed into SSE events (progress, claims, verification, ...).
    The SSEStatsHandler additionally pushes tool/LLM activity as it happens.
    """
    queue = _job_events[job_id]
    handler = SSEStatsHandler(loop, queue)

    ta = TradingAgentsGraph(
        selected_analysts=config.get(
            "selected_analysts",
            ["fundamentals", "technical", "game_theory", "news_sentiment"],
        ),
        config=config,
        callbacks=[handler],
    )

    init_state = ta.propagator.create_initial_state(ticker, date)
    args = ta.propagator.get_graph_args(callbacks=[handler])
    # Dual mode: "updates" gives per-node completion (partial state writes),
    # "messages" gives every message with (langgraph_node, run_id) so LLM
    # calls can be attributed to the graph node that made them.
    args["stream_mode"] = ["updates", "messages"]

    _push_event(loop, queue, {"event": "analysis_started", "ticker": ticker, "date": date})

    merged: Dict[str, Any] = {}
    seen_reports: Dict[str, str] = {}
    seen_verification: Dict[str, int] = {}
    node_stats: Dict[str, Dict[str, Any]] = {}
    node_started: Dict[str, float] = {}
    last_event_ts: float = time.time()

    def _add_node_token(node: str, tokens: Dict[str, int], duration: Optional[float]) -> None:
        stats = node_stats.setdefault(node, {
            "calls": 0, "tokens_in": 0, "tokens_out": 0, "time_ms": 0.0,
        })
        stats["calls"] += 1
        stats["tokens_in"] += tokens.get("in", 0) if tokens else 0
        stats["tokens_out"] += tokens.get("out", 0) if tokens else 0
        if duration is not None:
            stats["time_ms"] += duration * 1000

    for mode, data in ta.graph.stream(init_state, **args):
        if mode == "updates":
            for node, update in data.items():
                now = time.time()
                if node not in node_started:
                    node_started[node] = now
                update = update or {}  # nodes that wrote nothing emit None
                for event in _extract_update_events(node, update, merged, seen_reports, seen_verification):
                    if event["event"] == "node":
                        # per-node wall time: time since the previous node event
                        node_stats.setdefault(node, {
                            "calls": 0, "tokens_in": 0, "tokens_out": 0, "time_ms": 0.0,
                        })
                        node_stats[node]["time_ms"] += (now - last_event_ts) * 1000
                        event["node_stats"] = dict(node_stats)
                        last_event_ts = now
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
            msg, meta = data
            node = (meta or {}).get("langgraph_node", "")
            run_id = getattr(msg, "run_id", None)
            rec = handler.consume_record(run_id)
            if rec is not None:
                tokens = rec.get("tokens") or {}
                output = rec.get("output") or _excerpt(getattr(msg, "content", ""), 600)
                duration = rec.get("duration")
                _add_node_token(node, tokens, duration)
                _push_event(loop, queue, {
                    "event": "llm_call",
                    "node": node,
                    "input": rec.get("input", ""),
                    "output": output,
                    "tokens": tokens,
                    "duration": duration,
                    "node_stats": {
                        k: dict(v) for k, v in node_stats.items()
                    },
                })
            _push_event(loop, queue, {"event": "stats", "stats": handler.get_stats()})

    _push_event(loop, queue, {
        "event": "completed",
        "decision": _extract_decision(merged),
        "final_trade_decision": merged.get("final_trade_decision", ""),
        "stats": handler.get_stats(),
        "node_stats": {k: dict(v) for k, v in node_stats.items()},
    })
    return merged


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

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = os.environ.get("WEB_LLM_PROVIDER", "deepseek")
    config["quick_think_llm"] = os.environ.get("WEB_QUICK_MODEL", "deepseek-v4-flash")
    config["deep_think_llm"] = os.environ.get("WEB_DEEP_MODEL", "deepseek-v4-flash")
    config["max_debate_rounds"] = req.depth
    config["max_risk_discuss_rounds"] = req.depth
    config["output_language"] = req.language
    config["checkpoint_enabled"] = req.checkpoint
    config["selected_analysts"] = req.analysts
    config["web_search_enabled"] = (
        os.environ.get("WEB_SEARCH_ENABLED", "false").lower() == "true"
    )

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "ticker": req.ticker,
        "date": req.date,
        "created_at": now,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    _job_events[job_id] = asyncio.Queue()

    # Run the (synchronous) analysis in a dedicated thread. The SSE events are
    # marshalled into the asyncio queue via call_soon_threadsafe, so this works
    # both under uvicorn and under sync test clients (an asyncio.create_task
    # would never be scheduled in the latter).
    loop = asyncio.get_running_loop()

    def _run_sync():
        _jobs[job_id]["status"] = "running"
        try:
            final_state = _run_analysis_sync(loop, job_id, req.ticker, req.date, config)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = datetime.now().isoformat()
            _jobs[job_id]["result"] = {
                "decision": _extract_decision(final_state),
                "reports": _build_reports(final_state),
                "verification": final_state.get("verification_state", {}),
                "full_state": _serialize_state(final_state),
            }
        except Exception as exc:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
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
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=600.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("event") in ("completed", "error"):
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
