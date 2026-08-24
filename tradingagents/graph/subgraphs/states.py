"""Per-stage state schemas for the multi-subgraph trading pipeline.

Each stage subgraph declares only the channels it actually reads or writes.
The parent graph (AgentState) hands its full state to every subgraph node;
LangGraph filters it down to the subgraph's declared channels, runs the
stage, then merges the written channels back into the parent state.

Keeping every schema a strict subset of AgentState means:

- each stage is independently testable with a tiny fake state,
- no stage can accidentally reach into another stage's private data,
- the parent graph stays a thin sequential orchestrator, and
- checkpoints taken inside any subgraph remain resumable from the parent.
"""

from __future__ import annotations

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
    merge_verification_state,
)

# ---------------------------------------------------------------------------
# Stage 1: Analyst Team and its independently compiled analyst branches
# ---------------------------------------------------------------------------


class AnalystSubgraphSharedState(TypedDict):
    """Read-only inputs shared by every analyst branch.

    ``verification_state`` is also readable inside a branch so a retry can
    inject the latest fact-check feedback into the analyst prompt.  Branch
    output schemas decide whether that channel is written back to the team.
    """

    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    verification_state: Annotated[dict, merge_verification_state]


class FundamentalsAnalystChannels(TypedDict):
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst (基本面)"]
    fundamentals_claims: Annotated[list, "Structured claims from the Fundamentals Analyst"]
    messages_fundamentals: Annotated[list, add_messages]


class TechnicalAnalystChannels(TypedDict):
    technical_report: Annotated[str, "Report from the Technical Analyst (技术面)"]
    technical_claims: Annotated[list, "Structured claims from the Technical Analyst"]
    messages_technical: Annotated[list, add_messages]


class GameTheoryAnalystChannels(TypedDict):
    game_theory_report: Annotated[str, "Report from the Game Theory Analyst (博弈面)"]
    game_theory_claims: Annotated[list, "Structured claims from the Game Theory Analyst"]
    messages_game_theory: Annotated[list, add_messages]


class NewsSentimentAnalystChannels(TypedDict):
    news_sentiment_report: Annotated[str, "Report from the News & Sentiment Analyst (新闻舆情)"]
    messages_news_sentiment: Annotated[list, add_messages]


class MacroAnalystChannels(TypedDict):
    macro_environment_report: Annotated[str, "Report from the Macro/Market Environment Analyst (宏观环境)"]
    messages_macro: Annotated[list, add_messages]


class FundamentalsAnalystSubgraphState(
    AnalystSubgraphSharedState,
    FundamentalsAnalystChannels,
):
    """Private state of the compiled fundamentals analyst branch."""


class TechnicalAnalystSubgraphState(
    AnalystSubgraphSharedState,
    TechnicalAnalystChannels,
):
    """Private state of the compiled technical analyst branch."""


class GameTheoryAnalystSubgraphState(
    AnalystSubgraphSharedState,
    GameTheoryAnalystChannels,
):
    """Private state of the compiled game-theory analyst branch."""


class NewsSentimentAnalystSubgraphState(
    AnalystSubgraphSharedState,
    NewsSentimentAnalystChannels,
):
    """Private state of the compiled news/sentiment analyst branch."""


class MacroAnalystSubgraphState(
    AnalystSubgraphSharedState,
    MacroAnalystChannels,
):
    """Private state of the compiled macro/market-environment analyst branch."""


# A compiled branch must expose only the channels it owns.  If shared input
# fields such as company_of_interest were returned by all parallel branches,
# LangGraph would correctly reject the concurrent writes.  These output
# schemas make the fan-in explicit and collision-free.
class FundamentalsAnalystSubgraphOutput(FundamentalsAnalystChannels):
    verification_state: Annotated[dict, merge_verification_state]


class TechnicalAnalystSubgraphOutput(TechnicalAnalystChannels):
    verification_state: Annotated[dict, merge_verification_state]


class GameTheoryAnalystSubgraphOutput(GameTheoryAnalystChannels):
    verification_state: Annotated[dict, merge_verification_state]


class NewsSentimentAnalystSubgraphOutput(NewsSentimentAnalystChannels):
    """News currently has no LLM fact-checker, so it writes no verification state."""


class MacroAnalystSubgraphOutput(MacroAnalystChannels):
    """Macro has no LLM fact-checker/retry, so it writes no verification state."""


class AnalystTeamSubgraphState(MessagesState):
    """Compiled analyst subgraphs run in parallel (fundamentals / technical /
    game_theory / news_sentiment / macro).

    Channels written: the analyst reports (flow back to the parent so
    the Research Debate and Risk Debate stages can read them).
    """

    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst (基本面)"]
    technical_report: Annotated[str, "Report from the Technical Analyst (技术面)"]
    game_theory_report: Annotated[str, "Report from the Game Theory Analyst (博弈面)"]
    news_sentiment_report: Annotated[str, "Report from the News & Sentiment Analyst (新闻舆情)"]
    macro_environment_report: Annotated[str, "Report from the Macro/Market Environment Analyst (宏观环境)"]
    fundamentals_claims: Annotated[list, "Structured claims with sources from the Fundamentals Analyst"]
    technical_claims: Annotated[list, "Structured claims with sources from the Technical Analyst"]
    game_theory_claims: Annotated[list, "Structured claims with sources from the Game Theory Analyst"]
    verification_state: Annotated[dict, merge_verification_state]
    # Per-analyst parallel message channels (analyst <-> tools loops)
    messages_fundamentals: Annotated[list, add_messages]
    messages_technical: Annotated[list, add_messages]
    messages_game_theory: Annotated[list, add_messages]
    messages_news_sentiment: Annotated[list, add_messages]
    messages_macro: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Stage 2: Research Debate (bull vs bear + Research Manager)
# ---------------------------------------------------------------------------


class ResearchDebateSubgraphState(MessagesState):
    """Bull/bear researchers trade arguments; Research Manager synthesises
    the debate into an investment_plan."""

    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst (基本面)"]
    technical_report: Annotated[str, "Report from the Technical Analyst (技术面)"]
    game_theory_report: Annotated[str, "Report from the Game Theory Analyst (博弈面)"]
    news_sentiment_report: Annotated[str, "Report from the News & Sentiment Analyst (新闻舆情)"]
    macro_environment_report: Annotated[str, "Report from the Macro/Market Environment Analyst (宏观环境)"]
    investment_debate_state: Annotated[InvestDebateState, "Bull/bear debate progress"]
    investment_plan: Annotated[str, "Plan generated by the Research Manager"]


# ---------------------------------------------------------------------------
# Stage 3: Trader
# ---------------------------------------------------------------------------


class TraderSubgraphState(MessagesState):
    """Turns the Research Manager's investment plan into a concrete
    transaction proposal (trader_investment_plan)."""

    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    game_theory_report: Annotated[str, "Report from the Game Theory Analyst (博弈面)"]
    news_sentiment_report: Annotated[str, "Report from the News & Sentiment Analyst (新闻舆情)"]
    investment_plan: Annotated[str, "Plan generated by the Research Manager"]
    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]
    sender: Annotated[str, "Agent that sent this message"]


# ---------------------------------------------------------------------------
# Stage 4: Risk Debate (aggressive / conservative / neutral)
# ---------------------------------------------------------------------------


class RiskDebateSubgraphState(MessagesState):
    """The three risk analysts debate the Trader's proposal until the debate
    budget is exhausted; the subgraph then hands control back to the parent."""

    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst (基本面)"]
    technical_report: Annotated[str, "Report from the Technical Analyst (技术面)"]
    game_theory_report: Annotated[str, "Report from the Game Theory Analyst (博弈面)"]
    news_sentiment_report: Annotated[str, "Report from the News & Sentiment Analyst (新闻舆情)"]
    macro_environment_report: Annotated[str, "Report from the Macro/Market Environment Analyst (宏观环境)"]
    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]
    risk_debate_state: Annotated[RiskDebateState, "Risk debate progress"]


# ---------------------------------------------------------------------------
# Stage 5: Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioManagerSubgraphState(MessagesState):
    """Portfolio Manager renders the final, non-delegable decision
    (final_trade_decision) from the risk debate + both plans."""

    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    investment_plan: Annotated[str, "Plan generated by the Research Manager"]
    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]
    risk_debate_state: Annotated[RiskDebateState, "Risk debate progress"]
    past_context: Annotated[str, "Memory log context injected at run start"]
    final_trade_decision: Annotated[str, "Final decision made by the Portfolio Manager"]
