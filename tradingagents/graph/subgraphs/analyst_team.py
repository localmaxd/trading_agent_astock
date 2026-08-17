"""Stage 1 subgraph: the Analyst Team.

Runs the selected analysts in sequence. Each analyst loops with its own
tool node (analyst -> tools -> analyst) until it stops calling tools, then
a message-clear node wipes the conversation and hands over to the next
analyst. The subgraph ends after the last analyst's message-clear, with the
four reports written back into the parent state.

Node names inside this subgraph ("Fundamentals Analyst", "tools_*",
"Msg Clear *") match the routing keys emitted by
ConditionalLogic.should_continue_<analyst> so the conditional logic is
reused unchanged.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from tradingagents.agents import (
    create_fundamentals_analyst,
    create_game_theory_analyst,
    create_news_sentiment_analyst,
    create_technical_analyst,
    create_msg_delete,
)

from .states import AnalystTeamSubgraphState

_ANALYST_CREATORS: Dict[str, Any] = {
    "fundamentals": create_fundamentals_analyst,
    "technical": create_technical_analyst,
    "game_theory": create_game_theory_analyst,
    "news_sentiment": create_news_sentiment_analyst,
}

_ANALYST_DISPLAY_NAMES: Dict[str, str] = {
    "fundamentals": "Fundamentals",
    "technical": "Technical",
    "game_theory": "Game_Theory",
    "news_sentiment": "News_Sentiment",
}


def analyst_display_name(analyst_type: str) -> str:
    """Map internal analyst type to the display name used for graph nodes."""
    return _ANALYST_DISPLAY_NAMES.get(analyst_type, analyst_type.capitalize())


def build_analyst_team_subgraph(
    quick_thinking_llm: Any,
    tool_nodes: Dict[str, Any],
    conditional_logic: Any,
    selected_analysts: list,
) -> Any:
    """Build and compile the Analyst Team subgraph.

    Args:
        quick_thinking_llm: LLM used by the analyst agents.
        tool_nodes: Mapping analyst_type -> ToolNode (created by TradingAgentsGraph).
        conditional_logic: ConditionalLogic instance providing the
            should_continue_<analyst> routing functions.
        selected_analysts: Analyst types to include (unknown types are
            silently skipped, matching the legacy behaviour).

    Returns:
        A compiled StateGraph embedding the analyst pipeline.
    """
    selected = [a for a in selected_analysts if a in _ANALYST_CREATORS]
    if len(selected) == 0:
        raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

    workflow = StateGraph(AnalystTeamSubgraphState)

    # Analyst + message-clear + tools nodes for each selected analyst
    for analyst_type in selected:
        display_name = analyst_display_name(analyst_type)
        creator = _ANALYST_CREATORS[analyst_type]
        workflow.add_node(f"{display_name} Analyst", creator(quick_thinking_llm))
        workflow.add_node(f"Msg Clear {display_name}", create_msg_delete())
        workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

    # Chain the analysts in sequence, each looping with its own tools
    workflow.add_edge(START, f"{analyst_display_name(selected[0])} Analyst")

    for i, analyst_type in enumerate(selected):
        display_name = analyst_display_name(analyst_type)
        analyst = f"{display_name} Analyst"
        tools = f"tools_{analyst_type}"
        clear = f"Msg Clear {display_name}"

        # analyst -> tools -> analyst while tool calls remain, otherwise clear
        workflow.add_conditional_edges(
            analyst,
            getattr(conditional_logic, f"should_continue_{analyst_type}"),
            [tools, clear],
        )
        workflow.add_edge(tools, analyst)

        # Hand over to the next analyst, or finish the stage after the last one
        if i < len(selected) - 1:
            nxt = analyst_display_name(selected[i + 1])
            workflow.add_edge(clear, f"{nxt} Analyst")
        else:
            workflow.add_edge(clear, END)

    return workflow.compile()
