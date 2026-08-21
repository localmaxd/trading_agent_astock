# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState

# Analyst type -> its parallel message channel (analysts run concurrently in
# the Analyst Team stage, each with its own conversation).
_ANALYST_MESSAGE_CHANNELS = {
    "fundamentals": "messages_fundamentals",
    "technical": "messages_technical",
    "game_theory": "messages_game_theory",
    "news_sentiment": "messages_news_sentiment",
}


def _analyst_messages(state: AgentState, analyst_type: str) -> list:
    """Messages of one analyst's parallel conversation (fallback: shared)."""
    channel = _ANALYST_MESSAGE_CHANNELS.get(analyst_type, "messages")
    messages = state.get(channel)
    if not messages:
        messages = state.get("messages", [])
    return messages


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = _analyst_messages(state, "fundamentals")
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "done"

    def should_continue_technical(self, state: AgentState):
        """Determine if technical analysis should continue."""
        messages = _analyst_messages(state, "technical")
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_technical"
        return "done"

    def should_continue_game_theory(self, state: AgentState):
        """Determine if game theory analysis should continue."""
        messages = _analyst_messages(state, "game_theory")
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_game_theory"
        return "done"

    def should_continue_news_sentiment(self, state: AgentState):
        """Determine if news sentiment analysis should continue."""
        messages = _analyst_messages(state, "news_sentiment")
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news_sentiment"
        return "done"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
