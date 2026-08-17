"""Stage 3 subgraph: the Trader.

A single node stage: the Trader reads the Research Manager's investment
plan (plus the game-theory and news-sentiment reports), consults the
position tool, and produces a concrete transaction proposal
(trader_investment_plan).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents import create_trader

from .states import TraderSubgraphState


def build_trader_subgraph(quick_thinking_llm: Any) -> Any:
    """Build and compile the Trader subgraph.

    Args:
        quick_thinking_llm: LLM used by the Trader.

    Returns:
        A compiled StateGraph with a single Trader node.
    """
    workflow = StateGraph(TraderSubgraphState)

    workflow.add_node("Trader", create_trader(quick_thinking_llm))
    workflow.add_edge(START, "Trader")
    workflow.add_edge("Trader", END)

    return workflow.compile()
