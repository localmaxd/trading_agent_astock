"""Stage 4 subgraph: the risk management debate.

The aggressive, conservative and neutral analysts debate the Trader's
proposal in a fixed rotation (Aggressive -> Conservative -> Neutral ->
Aggressive ...), governed by ConditionalLogic.should_continue_risk_analysis.
When the debate budget is exhausted the routing key "Portfolio Manager" is
mapped to END, handing control back to the parent graph which then runs the
real Portfolio Manager stage.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents import (
    create_aggressive_debator,
    create_conservative_debator,
    create_neutral_debator,
)

from .states import RiskDebateSubgraphState


def build_risk_debate_subgraph(
    quick_thinking_llm: Any,
    conditional_logic: Any,
) -> Any:
    """Build and compile the Risk Debate subgraph.

    Args:
        quick_thinking_llm: LLM used by the three risk debators.
        conditional_logic: ConditionalLogic instance providing
            should_continue_risk_analysis.

    Returns:
        A compiled StateGraph embedding the risk debate pipeline.
    """
    workflow = StateGraph(RiskDebateSubgraphState)

    workflow.add_node("Aggressive Analyst", create_aggressive_debator(quick_thinking_llm))
    workflow.add_node("Conservative Analyst", create_conservative_debator(quick_thinking_llm))
    workflow.add_node("Neutral Analyst", create_neutral_debator(quick_thinking_llm))

    workflow.add_edge(START, "Aggressive Analyst")

    # The conditional logic's terminal routing key is "Portfolio Manager";
    # inside this stage that maps to END (the parent runs the PM subgraph).
    workflow.add_conditional_edges(
        "Aggressive Analyst",
        conditional_logic.should_continue_risk_analysis,
        {
            "Conservative Analyst": "Conservative Analyst",
            "Portfolio Manager": END,
        },
    )
    workflow.add_conditional_edges(
        "Conservative Analyst",
        conditional_logic.should_continue_risk_analysis,
        {
            "Neutral Analyst": "Neutral Analyst",
            "Portfolio Manager": END,
        },
    )
    workflow.add_conditional_edges(
        "Neutral Analyst",
        conditional_logic.should_continue_risk_analysis,
        {
            "Aggressive Analyst": "Aggressive Analyst",
            "Portfolio Manager": END,
        },
    )

    return workflow.compile()
