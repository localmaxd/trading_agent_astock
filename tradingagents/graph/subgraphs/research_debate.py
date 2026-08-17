"""Stage 2 subgraph: the bull vs bear research debate.

Bull and bear researchers trade arguments back and forth, one node per
turn, governed by ConditionalLogic.should_continue_debate (count-based
budget with speaker-based alternation). When the debate budget is
exhausted the Research Manager synthesises the whole conversation into an
investment_plan and the stage ends.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from tradingagents.agents import (
    create_bear_researcher,
    create_bull_researcher,
    create_research_manager,
)

from .states import ResearchDebateSubgraphState


def build_research_debate_subgraph(
    quick_thinking_llm: Any,
    deep_thinking_llm: Any,
    conditional_logic: Any,
) -> Any:
    """Build and compile the Research Debate subgraph.

    Args:
        quick_thinking_llm: LLM used by the bull/bear researchers.
        deep_thinking_llm: LLM used by the Research Manager.
        conditional_logic: ConditionalLogic instance providing
            should_continue_debate.

    Returns:
        A compiled StateGraph embedding the debate pipeline.
    """
    workflow = StateGraph(ResearchDebateSubgraphState)

    workflow.add_node("Bull Researcher", create_bull_researcher(quick_thinking_llm))
    workflow.add_node("Bear Researcher", create_bear_researcher(quick_thinking_llm))
    workflow.add_node("Research Manager", create_research_manager(deep_thinking_llm))

    workflow.add_edge(START, "Bull Researcher")

    workflow.add_conditional_edges(
        "Bull Researcher",
        conditional_logic.should_continue_debate,
        {
            "Bear Researcher": "Bear Researcher",
            "Research Manager": "Research Manager",
        },
    )
    workflow.add_conditional_edges(
        "Bear Researcher",
        conditional_logic.should_continue_debate,
        {
            "Bull Researcher": "Bull Researcher",
            "Research Manager": "Research Manager",
        },
    )

    workflow.add_edge("Research Manager", END)

    return workflow.compile()
