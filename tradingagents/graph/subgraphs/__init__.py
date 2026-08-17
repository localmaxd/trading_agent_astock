"""Multi-subgraph decomposition of the trading pipeline.

The full pipeline is split into five independently compiled stage subgraphs
orchestrated by a thin parent graph:

    Analyst Team -> Research Debate -> Trader -> Risk Debate -> Portfolio Manager

Each builder returns a compiled StateGraph whose state schema is a strict
subset of AgentState (see states.py). The parent graph embeds them as
nodes and routes sequentially; LangGraph merges each subgraph's written
channels back into the parent state.
"""

from .states import (
    AnalystTeamSubgraphState,
    PortfolioManagerSubgraphState,
    ResearchDebateSubgraphState,
    RiskDebateSubgraphState,
    TraderSubgraphState,
)
from .analyst_team import analyst_display_name, build_analyst_team_subgraph
from .research_debate import build_research_debate_subgraph
from .trader import build_trader_subgraph
from .risk_debate import build_risk_debate_subgraph
from .portfolio_manager import build_portfolio_manager_subgraph

__all__ = [
    "AnalystTeamSubgraphState",
    "ResearchDebateSubgraphState",
    "TraderSubgraphState",
    "RiskDebateSubgraphState",
    "PortfolioManagerSubgraphState",
    "analyst_display_name",
    "build_analyst_team_subgraph",
    "build_research_debate_subgraph",
    "build_trader_subgraph",
    "build_risk_debate_subgraph",
    "build_portfolio_manager_subgraph",
]
