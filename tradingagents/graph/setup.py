# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_states import AgentState

from .subgraphs import (
    build_analyst_team_subgraph,
    build_portfolio_manager_subgraph,
    build_research_debate_subgraph,
    build_risk_debate_subgraph,
    build_trader_subgraph,
)
from .ticker_guard import create_ticker_guard


class GraphSetup:
    """Builds the multi-subgraph trading workflow.

    The pipeline is decomposed into five independently compiled stage
    subgraphs, orchestrated by a thin parent graph:

        Analyst Team -> Research Debate -> Trader -> Risk Debate -> Portfolio Manager

    Each stage subgraph declares only the state channels it needs (a strict
    subset of AgentState, see tradingagents.graph.subgraphs.states). The
    parent graph passes its full AgentState into every stage node, LangGraph
    filters it to the stage's schema, and merges the written channels back
    afterwards — so the stages stay decoupled while the public
    TradingAgentsGraph API (propagate/stream/resume/checkpoints) is
    unchanged.
    """

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: Any,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(
        self,
        selected_analysts=["fundamentals", "technical", "game_theory", "news_sentiment"],
    ):
        """Set up the parent agent workflow graph from stage subgraphs.

        Args:
            selected_analysts (list): List of analyst types to include in the
                Analyst Team stage. Options are:
                - "fundamentals": 基本面 Analyst
                - "technical": 技术面 Analyst
                - "game_theory": 博弈面 Analyst
                - "news_sentiment": 新闻舆情 Analyst
                Unknown types are silently skipped (legacy behaviour).

        Returns:
            A (uncompiled) StateGraph over the full AgentState whose nodes are
            the five compiled stage subgraphs.
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # --- Stage subgraphs (each compiled independently) ---
        analyst_team = build_analyst_team_subgraph(
            quick_thinking_llm=self.quick_thinking_llm,
            tool_nodes=self.tool_nodes,
            conditional_logic=self.conditional_logic,
            selected_analysts=selected_analysts,
            # The fact-checker does the careful re-verification work, so it
            # gets the deep-thinking LLM when one is configured.
            verify_llm=self.deep_thinking_llm,
        )
        research_debate = build_research_debate_subgraph(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_thinking_llm=self.deep_thinking_llm,
            conditional_logic=self.conditional_logic,
        )
        trader = build_trader_subgraph(quick_thinking_llm=self.quick_thinking_llm)
        risk_debate = build_risk_debate_subgraph(
            quick_thinking_llm=self.quick_thinking_llm,
            conditional_logic=self.conditional_logic,
        )
        portfolio_manager = build_portfolio_manager_subgraph(
            deep_thinking_llm=self.deep_thinking_llm,
        )

        # --- Parent graph: thin sequential orchestrator ---
        # Every stage is preceded by a TickerGuard that verifies the
        # instrument under analysis (format + anchor consistency) and
        # normalises it before the stage runs.
        workflow = StateGraph(AgentState)

        # NOTE: ':' is a reserved character in LangGraph node names, so guards
        # are named with '-' instead.
        guard_names = {
            "Analyst Team": "TickerGuard-Analyst Team",
            "Research Debate": "TickerGuard-Research Debate",
            "Trader": "TickerGuard-Trader",
            "Risk Debate": "TickerGuard-Risk Debate",
            "Portfolio Manager": "TickerGuard-Portfolio Manager",
        }
        for stage_name, guard_name in guard_names.items():
            workflow.add_node(guard_name, create_ticker_guard(stage_name))

        workflow.add_node("Analyst Team", analyst_team)
        workflow.add_node("Research Debate", research_debate)
        workflow.add_node("Trader", trader)
        workflow.add_node("Risk Debate", risk_debate)
        workflow.add_node("Portfolio Manager", portfolio_manager)

        workflow.add_edge(START, guard_names["Analyst Team"])
        workflow.add_edge(guard_names["Analyst Team"], "Analyst Team")
        workflow.add_edge("Analyst Team", guard_names["Research Debate"])
        workflow.add_edge(guard_names["Research Debate"], "Research Debate")
        workflow.add_edge("Research Debate", guard_names["Trader"])
        workflow.add_edge(guard_names["Trader"], "Trader")
        workflow.add_edge("Trader", guard_names["Risk Debate"])
        workflow.add_edge(guard_names["Risk Debate"], "Risk Debate")
        workflow.add_edge("Risk Debate", guard_names["Portfolio Manager"])
        workflow.add_edge(guard_names["Portfolio Manager"], "Portfolio Manager")
        workflow.add_edge("Portfolio Manager", END)

        return workflow
