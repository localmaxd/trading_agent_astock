"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import json

from langchain_core.messages import AIMessage, ToolMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.external_api_tools import position

from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")
    tools = [position]

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        game_theory_report = state["game_theory_report"]
        news_sentiment_report = state["news_sentiment_report"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a disciplined trading agent. Based on market data, analysts' reports, "
                    "please use tool 'position' to get the available_capital, total_shares, market_value "
                    "of the current ts_code before making your decision. "
                    "Then based on the investment plan, game theory report, and news sentiment report, "
                    "you must output a concrete trading decision. "
                    "Your response must include:\n"
                    "1. Action: [Buy, Sell, Hold]\n"
                    "2. If Action is Buy or Sell:\n"
                    "   - Position Size: (e.g., '15% of portfolio', '200 shares')\n"
                    "   - Entry Timing: (e.g., 'Market order now', 'Limit order at $152.30')\n"
                    "   - Stop-Loss: (e.g., '$147.50', '-3% from entry')\n"
                    "   - Add-on Strategy: (conditions to increase position. If none, state 'None')\n"
                    "3. Justification: anchor your reasoning in the analysts' reports and the position data.\n"
                    "Be precise and actionable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context}\n\n"
                    f"--- Game Theory Report ---\n{game_theory_report}\n\n"
                    f"--- News Sentiment Report ---\n{news_sentiment_report}\n\n"
                    f"Proposed Investment Plan: {investment_plan}\n\n"
                    f"First call the 'position' tool with ts_code={company_name} to check available "
                    f"capital, current holdings, and market value. Then make your trading decision."
                ),
            },
        ]

        # Step 1: Let the LLM call the position tool
        tool_llm = llm.bind_tools(tools)
        tool_response = tool_llm.invoke(messages)

        # Step 2: Execute tool calls and append results to messages
        if hasattr(tool_response, "tool_calls") and tool_response.tool_calls:
            for tc in tool_response.tool_calls:
                tool_name = tc.get("name") if isinstance(tc, dict) else tc.name
                tool_args = tc.get("args") if isinstance(tc, dict) else tc.args
                if tool_name == "position":
                    ts_code = tool_args.get("ts_code", company_name)
                    result = position.invoke({"ts_code": ts_code})
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc] if isinstance(tc, dict) else [{
                            "id": getattr(tc, "id", "call_1"),
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "call_1"),
                        "content": result,
                    })

        # Step 3: Generate the structured trading proposal
        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
