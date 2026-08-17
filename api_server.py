"""TradingAgents FastAPI server with async execution and SSE streaming.

Endpoints:
  POST /api/v1/analyze          → Start analysis, return job_id
  GET  /api/v1/analyze/{job_id}/stream  → SSE stream of progress
  GET  /api/v1/analyze/{job_id}/result  → Final result
"""

from __future__ import annotations

import asyncio
import logging
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

app = FastAPI(
    title="TradingAgents API",
    description="Multi-agent LLM financial trading framework",
    version="0.2.4",
)

# CORS: allow all origins
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

# Thread pool for running sync TradingAgentsGraph
_executor = ThreadPoolExecutor(max_workers=4)

# Analyst name mapping for display
_ANALYST_NAMES = {
    "fundamentals_report": "Fundamentals Analyst (基本面)",
    "technical_report": "Technical Analyst (技术面)",
    "game_theory_report": "Game Theory Analyst (博弈面)",
    "risk_report": "Risk Analyst (风险面)",
    "news_sentiment_report": "News Sentiment Analyst (新闻舆情)",
}

_ANALYST_ORDER = [
    "fundamentals_report", "technical_report", "game_theory_report",
    "risk_report", "news_sentiment_report",
]


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")
    date: str = Field(..., description="Analysis date in YYYY-MM-DD format")
    analysts: List[str] = Field(
        default=["fundamentals", "technical", "game_theory", "risk", "news_sentiment"],
        description="List of analyst types to include (基本面/技术面/博弈面/风险面/新闻舆情)",
    )
    depth: int = Field(default=1, ge=1, le=5, description="Research depth (debate rounds)")
    language: str = Field(default="Chinese", description="Output language for reports")
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
    error: Optional[str] = None


def _extract_progress(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract meaningful progress events from a graph stream chunk."""
    events = []

    # Analyst reports completed
    for key, name in _ANALYST_NAMES.items():
        if chunk.get(key) and chunk[key].strip():
            events.append({
                "event": "agent_completed",
                "agent": name,
                "report_key": key,
                "report": chunk[key],
            })

    # Investment debate progress
    if chunk.get("investment_debate_state"):
        debate = chunk["investment_debate_state"]
        if debate.get("judge_decision") and debate["judge_decision"].strip():
            events.append({
                "event": "agent_completed",
                "agent": "Research Manager",
                "report_key": "investment_plan",
                "report": debate["judge_decision"],
            })

    # Trader plan
    if chunk.get("trader_investment_plan") and chunk["trader_investment_plan"].strip():
        events.append({
            "event": "agent_completed",
            "agent": "Trader",
            "report_key": "trader_investment_plan",
            "report": chunk["trader_investment_plan"],
        })

    # Risk debate + Portfolio Manager decision
    if chunk.get("risk_debate_state"):
        risk = chunk["risk_debate_state"]
        if risk.get("judge_decision") and risk["judge_decision"].strip():
            events.append({
                "event": "decision",
                "agent": "Portfolio Manager",
                "report_key": "final_trade_decision",
                "report": risk["judge_decision"],
            })

    return events


def _run_analysis_sync(
    loop: asyncio.AbstractEventLoop,
    job_id: str,
    ticker: str,
    date: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run TradingAgentsGraph in a thread, pushing events to the queue.

    Uses ta.propagate() so that _run_graph() -> _log_agent_progress()
    prints console logs for each agent step.
    """
    queue = _job_events[job_id]

    try:
        ta = TradingAgentsGraph(
            selected_analysts=config.get("selected_analysts", ["fundamentals", "technical", "game_theory", "risk", "news_sentiment"]),
            config=config,
        )

        # Use propagate() to go through _run_graph() which logs each step
        final_state, decision = ta.propagate(ticker, date)

        # Push extracted reports to SSE queue after completion
        reports = _build_reports(final_state)
        for agent_name, report in reports.items():
            if report and isinstance(report, str) and report.strip():
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "event": "agent_completed",
                        "agent": agent_name,
                        "report": report[:500] + "..." if len(report) > 500 else report,
                    }
                )
            elif report and isinstance(report, dict):
                # Nested reports (research/risk teams)
                for sub_name, sub_report in report.items():
                    if sub_report and isinstance(sub_report, str) and sub_report.strip():
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "event": "agent_completed",
                                "agent": f"{agent_name} - {sub_name}",
                                "report": sub_report[:500] + "..." if len(sub_report) > 500 else sub_report,
                            }
                        )

        # Send completion event
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {
                "event": "completed",
                "decision": decision,
                "final_trade_decision": final_state.get("final_trade_decision", ""),
            }
        )

        return final_state

    except Exception as exc:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"event": "error", "error": str(exc)}
        )
        raise


def _build_reports(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Build structured reports from final state."""
    reports = {}

    # Analyst reports
    for key, name in _ANALYST_NAMES.items():
        if final_state.get(key):
            reports[name] = final_state[key]

    # Research team
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        reports["Research Team"] = {
            "bull": debate.get("bull_history", ""),
            "bear": debate.get("bear_history", ""),
            "manager_decision": debate.get("judge_decision", ""),
        }

    # Trader
    if final_state.get("trader_investment_plan"):
        reports["Trader"] = final_state["trader_investment_plan"]

    # Risk + Portfolio
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        reports["Risk Management"] = {
            "aggressive": risk.get("aggressive_history", ""),
            "conservative": risk.get("conservative_history", ""),
            "neutral": risk.get("neutral_history", ""),
        }
        reports["Portfolio Manager"] = risk.get("judge_decision", "")

    return reports


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """Start a new analysis job. Returns immediately with job_id."""
    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "deepseek"
    config["quick_think_llm"] = "deepseek-chat"
    config["deep_think_llm"] = "deepseek-chat"
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
    }
    _job_events[job_id] = asyncio.Queue()

    # Start analysis in background thread
    async def _background():
        _jobs[job_id]["status"] = "running"
        loop = asyncio.get_running_loop()
        try:
            final_state = await loop.run_in_executor(
                _executor,
                lambda: _run_analysis_sync(
                    loop, job_id, req.ticker, req.date, config
                ),
            )
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = datetime.now().isoformat()
            _jobs[job_id]["result"] = {
                "decision": _extract_decision(final_state),
                "reports": _build_reports(final_state),
                "full_state": _serialize_state(final_state),
            }
        except Exception as exc:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)

    asyncio.create_task(_background())

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
        # Send initial state
        yield f"data: {json.dumps({'event': 'started', 'job_id': job_id}, ensure_ascii=False)}\n\n"

        while True:
            try:
                # Wait for next event with timeout
                event = await asyncio.wait_for(queue.get(), timeout=300.0)
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
        error=job.get("error"),
    )


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all jobs (for debugging)."""
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
    # Keep only string/dict/list values
    result = {}
    for key, value in state.items():
        if key == "messages":
            # Skip LangChain message objects
            continue
        if isinstance(value, (str, int, float, bool, list, dict)):
            result[key] = value
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
