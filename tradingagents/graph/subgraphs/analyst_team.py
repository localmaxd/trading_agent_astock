"""Stage 1 subgraph: the parallel Analyst Team coordinator.

Each selected analyst is an independently compiled LangGraph subgraph.  The
team graph contains only those analyst subgraphs, fans them out from START,
and joins them at END.  Inside each analyst subgraph the original
analyst -> tools -> analyst loop and optional fact-check/retry loop are kept
unchanged.

The macro / market-environment analyst (only tool_market_environment, no
fact-check, no retry, no structured claims) is a fifth branch; fundamentals /
technical / game_theory additionally receive tool_schema so they can look up
field semantics of the (versioned) external API responses.

Each branch has a private state schema and a narrow output schema.  That is
important for true parallel execution: only the report, claims, messages and
verification entry owned by that analyst are merged back into the team, so
shared read-only inputs are never written concurrently.

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
    create_macro_analyst,
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
from .states import (
    AnalystTeamSubgraphState,
    FundamentalsAnalystSubgraphOutput,
    FundamentalsAnalystSubgraphState,
    GameTheoryAnalystSubgraphOutput,
    GameTheoryAnalystSubgraphState,
    MacroAnalystSubgraphOutput,
    MacroAnalystSubgraphState,
    NewsSentimentAnalystSubgraphOutput,
    NewsSentimentAnalystSubgraphState,
    TechnicalAnalystSubgraphOutput,
    TechnicalAnalystSubgraphState,
)

_ANALYST_CREATORS: Dict[str, Any] = {
    "fundamentals": create_fundamentals_analyst,
    "technical": create_technical_analyst,
    "game_theory": create_game_theory_analyst,
    "news_sentiment": create_news_sentiment_analyst,
    "macro": create_macro_analyst,
}

_ANALYST_DISPLAY_NAMES: Dict[str, str] = {
    "fundamentals": "Fundamentals",
    "technical": "Technical",
    "game_theory": "Game_Theory",
    "news_sentiment": "News_Sentiment",
    "macro": "Macro",
}

_ANALYST_STATE_SCHEMAS: Dict[str, Any] = {
    "fundamentals": FundamentalsAnalystSubgraphState,
    "technical": TechnicalAnalystSubgraphState,
    "game_theory": GameTheoryAnalystSubgraphState,
    "news_sentiment": NewsSentimentAnalystSubgraphState,
    "macro": MacroAnalystSubgraphState,
}

_ANALYST_OUTPUT_SCHEMAS: Dict[str, Any] = {
    "fundamentals": FundamentalsAnalystSubgraphOutput,
    "technical": TechnicalAnalystSubgraphOutput,
    "game_theory": GameTheoryAnalystSubgraphOutput,
    "news_sentiment": NewsSentimentAnalystSubgraphOutput,
    "macro": MacroAnalystSubgraphOutput,
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


def _build_analyst_subgraph(
    analyst_type: str,
    quick_thinking_llm: Any,
    tool_node: Any,
    conditional_logic: Any,
    verify_enabled: bool,
    verify_llm: Any,
    max_verify_rounds: int,
) -> Any:
    """Build one self-contained analyst/tool/verification branch.

    The compiled graph is embedded as one node in the Analyst Team graph.
    ``output_schema`` deliberately excludes shared inputs such as ticker and
    trade date; returning those from parallel nodes would be an invalid
    concurrent update in LangGraph.
    """
    display_name = analyst_display_name(analyst_type)
    creator = _ANALYST_CREATORS[analyst_type]
    extra_tools = [web_search_tool] if _web_search_enabled_for(analyst_type) else []
    analyst_node = "Analyze"
    tools_node = f"tools_{analyst_type}"

    workflow = StateGraph(
        _ANALYST_STATE_SCHEMAS[analyst_type],
        output_schema=_ANALYST_OUTPUT_SCHEMAS[analyst_type],
    )
    workflow.add_node(
        analyst_node,
        creator(quick_thinking_llm, extra_tools=extra_tools),
    )
    workflow.add_node(tools_node, tool_node)

    should_verify = verify_enabled and analyst_type in VERIFY_ANALYSTS
    if should_verify:
        checker = f"FactChecker-{display_name}"
        retry_clear = f"RetryClear-{display_name}"
        workflow.add_node(
            checker,
            create_fact_checker(analyst_type, max_rounds=max_verify_rounds),
        )
        workflow.add_node(
            retry_clear,
            create_msg_delete(messages_key=f"messages_{analyst_type}"),
        )

    workflow.add_edge(START, analyst_node)
    if should_verify:
        workflow.add_conditional_edges(
            analyst_node,
            getattr(conditional_logic, f"should_continue_{analyst_type}"),
            {tools_node: tools_node, "done": checker},
        )
        workflow.add_edge(tools_node, analyst_node)
        workflow.add_conditional_edges(
            checker,
            make_verify_router(analyst_type, max_verify_rounds),
            {"next": END, "retry": retry_clear},
        )
        workflow.add_edge(retry_clear, analyst_node)
    else:
        workflow.add_conditional_edges(
            analyst_node,
            getattr(conditional_logic, f"should_continue_{analyst_type}"),
            {tools_node: tools_node, "done": END},
        )
        workflow.add_edge(tools_node, analyst_node)

    return workflow.compile(name=f"{display_name} Analyst")


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
        A compiled coordinator graph containing one compiled subgraph per
        selected analyst.
    """
    cfg = get_config()
    verify_enabled = cfg.get("verify_enabled", True) and verify_llm is not None
    max_verify_rounds = int(cfg.get("max_verify_rounds", 2))

    selected = [a for a in selected_analysts if a in _ANALYST_CREATORS]
    if len(selected) == 0:
        raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

    workflow = StateGraph(AnalystTeamSubgraphState)

    # The team level contains only coarse-grained analyst subgraphs (one per
    # selected analyst type).  Tool, fact-check and retry nodes live inside
    # their owning branch.
    for analyst_type in selected:
        display_name = analyst_display_name(analyst_type)
        workflow.add_node(
            f"{display_name} Analyst",
            _build_analyst_subgraph(
                analyst_type=analyst_type,
                quick_thinking_llm=quick_thinking_llm,
                tool_node=tool_nodes[analyst_type],
                conditional_logic=conditional_logic,
                verify_enabled=verify_enabled,
                verify_llm=verify_llm,
                max_verify_rounds=max_verify_rounds,
            ),
        )

    # PARALLEL fan-out/fan-in. LangGraph waits for all selected compiled
    # subgraphs to finish before the Analyst Team stage returns.
    for analyst_type in selected:
        analyst_subgraph = f"{analyst_display_name(analyst_type)} Analyst"
        workflow.add_edge(START, analyst_subgraph)
        workflow.add_edge(analyst_subgraph, END)

    return workflow.compile(name="Analyst Team")
