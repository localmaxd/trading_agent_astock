# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config
from .ticker_guard import normalize_ticker

# Import external API tools for the 5 analysts
from tradingagents.agents.utils.external_api_tools import (
    tool_fundamental, tool_technical, tool_special_data,
    tool_game_theory, tool_risk, tool_news_sentiment,
)

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["fundamentals", "technical", "game_theory", "news_sentiment"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include.
                默认: ["fundamentals", "technical", "game_theory", "risk", "news_sentiment"]
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for the 4 analysts using external API tools."""
        return {
            "fundamentals": ToolNode([tool_fundamental]),
            "technical": ToolNode([tool_technical, tool_special_data]),
            "game_theory": ToolNode([tool_game_theory]),
            "news_sentiment": ToolNode([tool_news_sentiment]),
        }

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.
        For A-shares, uses CSI 300 (510300) as benchmark instead of SPY.
        """
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            benchmark = yf.Ticker("510300.SS").history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(benchmark) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(benchmark) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (benchmark["Close"].iloc[actual_days] - benchmark["Close"].iloc[0])
                / benchmark["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s (will retry next run): %s",
                ticker, trade_date, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run."""
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(ticker, entry["date"])
            if raw is None:
                continue
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""
        # Canonicalise the instrument up front (fail-fast on malformed input).
        # The TickerGuard nodes inside the graph enforce the same invariant,
        # so state, memory log, and saved reports all use one canonical code.
        company_name = normalize_ticker(company_name)
        self.ticker = company_name

        self._resolve_pending_entries(company_name)

        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _log_agent_progress(self, prev_state, curr_state):
        """Log which agent step just completed based on state changes."""
        import sys
        if prev_state is None:
            sys.stdout.write("[Agent Step] Analysis pipeline started\n")
            sys.stdout.flush()
            return

        # 4 analyst reports
        for key, name in [
            ("fundamentals_report", "Fundamentals Analyst (基本面)"),
            ("technical_report", "Technical Analyst (技术面)"),
            ("game_theory_report", "Game Theory Analyst (博弈面)"),
            ("news_sentiment_report", "News Sentiment Analyst (新闻舆情)"),
        ]:
            prev_val = prev_state.get(key, "")
            curr_val = curr_state.get(key, "")
            if curr_val and curr_val != prev_val and curr_val.strip():
                sys.stdout.write(f"[Agent Step] {name} analysis completed ({len(curr_val)} chars)\n")
                sys.stdout.flush()
                return

        # Research team debate
        prev_debate = prev_state.get("investment_debate_state") or {}
        curr_debate = curr_state.get("investment_debate_state") or {}
        if curr_debate.get("bull_history") != prev_debate.get("bull_history"):
            sys.stdout.write("[Agent Step] Bull Researcher debated\n")
            sys.stdout.flush()
            return
        if curr_debate.get("bear_history") != prev_debate.get("bear_history"):
            sys.stdout.write("[Agent Step] Bear Researcher debated\n")
            sys.stdout.flush()
            return
        if curr_debate.get("judge_decision") != prev_debate.get("judge_decision"):
            if curr_debate.get("judge_decision", "").strip():
                sys.stdout.write("[Agent Step] Research Manager synthesized investment plan\n")
                sys.stdout.flush()
                return

        # Trader
        prev_trader = prev_state.get("trader_investment_plan", "")
        curr_trader = curr_state.get("trader_investment_plan", "")
        if curr_trader and curr_trader != prev_trader and curr_trader.strip():
            sys.stdout.write("[Agent Step] Trader generated transaction proposal\n")
            sys.stdout.flush()
            return

        # Risk team debate
        prev_risk = prev_state.get("risk_debate_state") or {}
        curr_risk = curr_state.get("risk_debate_state") or {}
        if curr_risk.get("aggressive_history") != prev_risk.get("aggressive_history"):
            sys.stdout.write("[Agent Step] Aggressive Analyst debated\n")
            sys.stdout.flush()
            return
        if curr_risk.get("conservative_history") != prev_risk.get("conservative_history"):
            sys.stdout.write("[Agent Step] Conservative Analyst debated\n")
            sys.stdout.flush()
            return
        if curr_risk.get("neutral_history") != prev_risk.get("neutral_history"):
            sys.stdout.write("[Agent Step] Neutral Analyst debated\n")
            sys.stdout.flush()
            return
        if curr_risk.get("judge_decision") != prev_risk.get("judge_decision"):
            if curr_risk.get("judge_decision", "").strip():
                sys.stdout.write("[Agent Step] Portfolio Manager made final decision\n")
                sys.stdout.flush()
                return

    def _run_graph(self, company_name, trade_date):
        """Execute the graph and write the resulting state to disk and memory log."""
        past_context = self.memory_log.get_past_context(company_name)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context
        )
        args = self.propagator.get_graph_args()

        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        trace = []
        prev_state = None
        for chunk in self.graph.stream(init_agent_state, **args):
            import sys
            sys.stdout.write(
                f"[StreamChunk] msg_count={len(chunk.get('messages', []))} "
                f"fundamentals_len={len(chunk.get('fundamentals_report', ''))} "
                f"technical_len={len(chunk.get('technical_report', ''))} "
                f"game_theory_len={len(chunk.get('game_theory_report', ''))} "
                f"news_sentiment_len={len(chunk.get('news_sentiment_report', ''))} "
                f"trader_len={len(chunk.get('trader_investment_plan', ''))}\n"
            )
            sys.stdout.flush()
            self._log_agent_progress(prev_state, chunk)
            prev_state = chunk.copy()
            if self.debug:
                if len(chunk["messages"]) != 0:
                    chunk["messages"][-1].pretty_print()
            trace.append(chunk)
        final_state = trace[-1] if trace else init_agent_state

        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "fundamentals_report": final_state["fundamentals_report"],
            "technical_report": final_state["technical_report"],
            "game_theory_report": final_state["game_theory_report"],
            "news_sentiment_report": final_state["news_sentiment_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
