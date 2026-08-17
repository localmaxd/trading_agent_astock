"""Stage 5 subgraph: the Portfolio Manager.

A single node stage: the Portfolio Manager reads the risk debate, the
Research Manager's plan, the Trader's proposal, and any prior-decision
lessons from the memory log (past_context), then renders the final
non-delegable decision (final_trade_decision) via structured output.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents import create_portfolio_manager

from .states import PortfolioManagerSubgraphState


def build_portfolio_manager_subgraph(deep_thinking_llm: Any) -> Any:
    """Build and compile the Portfolio Manager subgraph.

    Args:
        deep_thinking_llm: LLM used by the Portfolio Manager.

    Returns:
        A compiled StateGraph with a single Portfolio Manager node.
    """
    workflow = StateGraph(PortfolioManagerSubgraphState)

    workflow.add_node("Portfolio Manager", create_portfolio_manager(deep_thinking_llm))
    workflow.add_edge(START, "Portfolio Manager")
    workflow.add_edge("Portfolio Manager", END)

    return workflow.compile()
