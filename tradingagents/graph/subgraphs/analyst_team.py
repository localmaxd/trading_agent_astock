"""Stage 1 subgraph: the Analyst Team.

The selected analysts run in PARALLEL: every analyst is fanned out from
START and loops with its own tool node (analyst -> tools -> analyst) until
it stops calling tools, then its branch finishes. Each branch owns a
dedicated message channel (messages_<analyst>) and writes its own report
channel, so the parallel conversations never contaminate each other and no
message-clearing is needed on the happy path. The subgraph ends when ALL
branches finish; the four reports are written back into the parent state.

Routing keys emitted by ConditionalLogic.should_continue_<analyst>:
"tools_*" (keep looping) or "done" (finish the branch).

Optional additions (config-driven, see default_config.py):

- **web search**: when web_search_enabled is True, the analysts listed in
  web_search_analysts also receive web_search_tool and may use it to fetch
  supplementary material (eastmoney.com etc.).
- **fact verification**: when verify_enabled is True, a FactChecker node
  follows the fundamentals / technical / game_theory outputs. On failure it
  stores feedback in verification_state and the router sends the analyst
  back (via a RetryClear node) to redo its material, up to max_verify_rounds
  attempts; beyond that the report is marked unverified and the pipeline
  continues.
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
from tradingagents.agents.utils.web_search_tool import web_search_tool
from tradingagents.dataflows.config import get_config

from ..fact_checker import (
    VERIFY_ANALYSTS,
    create_fact_checker,
    make_verify_router,
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


def _web_search_enabled_for(analyst_type: str) -> bool:
    """Whether web_search_tool should be offered to this analyst (config-driven)."""
    cfg = get_config()
    if not cfg.get("web_search_enabled", False):
        return False
    return analyst_type in (cfg.get("web_search_analysts") or [])


def build_analyst_team_subgraph(
    quick_thinking_llm: Any,
    tool_nodes: Dict[str, Any],
    conditional_logic: Any,
    selected_analysts: list,
    verify_llm: Any = None,
) -> Any:
    """Build and compile the Analyst Team subgraph.

    Args:
        quick_thinking_llm: LLM used by the analyst agents.
        tool_nodes: Mapping analyst_type -> ToolNode (created by TradingAgentsGraph).
        conditional_logic: ConditionalLogic instance providing the
            should_continue_<analyst> routing functions.
        selected_analysts: Analyst types to include (unknown types are
            silently skipped, matching the legacy behaviour).
        verify_llm: LLM used by the fact-checker nodes. When None the
            quick_thinking_llm is used. Verification is only added for
            analysts in VERIFY_ANALYSTS when config verify_enabled is True.

    Returns:
        A compiled StateGraph embedding the analyst pipeline.
    """
    cfg = get_config()
    verify_enabled = cfg.get("verify_enabled", True) and verify_llm is not None
    max_verify_rounds = int(cfg.get("max_verify_rounds", 2))

    selected = [a for a in selected_analysts if a in _ANALYST_CREATORS]
    if len(selected) == 0:
        raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

    workflow = StateGraph(AnalystTeamSubgraphState)

    # Analyst + tools (+ optional fact-checker) nodes. No message-clear
    # nodes on the happy path: each parallel branch has its own message
    # channel, so a finished analyst's conversation never leaks anywhere.
    for analyst_type in selected:
        display_name = analyst_display_name(analyst_type)
        creator = _ANALYST_CREATORS[analyst_type]
        extra_tools = [web_search_tool] if _web_search_enabled_for(analyst_type) else []
        workflow.add_node(f"{display_name} Analyst", creator(quick_thinking_llm, extra_tools=extra_tools))
        workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        if verify_enabled and analyst_type in VERIFY_ANALYSTS:
            workflow.add_node(
                f"FactChecker-{display_name}",
                create_fact_checker(analyst_type, verify_llm, max_verify_rounds),
            )
            workflow.add_node(
                f"RetryClear-{display_name}",
                create_msg_delete(messages_key=f"messages_{analyst_type}"),
            )

    # PARALLEL fan-out: every analyst starts from START simultaneously. Each
    # branch is an independent analyst <-> tools loop writing to its own
    # message channel and its own report channel, so the branches cannot
    # contaminate each other. The stage ends when ALL branches finish
    # (LangGraph joins the fan-out automatically at END).
    for analyst_type in selected:
        workflow.add_edge(START, f"{analyst_display_name(analyst_type)} Analyst")

    for analyst_type in selected:
        display_name = analyst_display_name(analyst_type)
        analyst = f"{display_name} Analyst"
        tools = f"tools_{analyst_type}"

        # analyst -> tools -> analyst while tool calls remain, otherwise the
        # branch finishes (fact-checker when enabled, else END).
        if verify_enabled and analyst_type in VERIFY_ANALYSTS:
            checker = f"FactChecker-{display_name}"
            retry_clear = f"RetryClear-{display_name}"
            workflow.add_conditional_edges(
                analyst,
                getattr(conditional_logic, f"should_continue_{analyst_type}"),
                {tools: tools, "done": checker},
            )
            workflow.add_edge(tools, analyst)
            workflow.add_conditional_edges(
                checker,
                make_verify_router(analyst_type, max_verify_rounds),
                {
                    "next": END,
                    "retry": retry_clear,
                },
            )
            # Retry path: wipe the analyst's own conversation, then it redoes
            # the report with the checker's feedback injected into its prompt.
            workflow.add_edge(retry_clear, analyst)
        else:
            workflow.add_conditional_edges(
                analyst,
                getattr(conditional_logic, f"should_continue_{analyst_type}"),
                {tools: tools, "done": END},
            )
            workflow.add_edge(tools, analyst)

    return workflow.compile()
