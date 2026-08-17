def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        fundamentals_report = state["fundamentals_report"]
        technical_report = state["technical_report"]
        game_theory_report = state["game_theory_report"]
        news_sentiment_report = state["news_sentiment_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""# Role Definition
You are a senior convexity trader (CFA/CAIA) specializing in **asymmetric payoff identification**. Your value function prioritizes expected value maximization, accepting that many small losses are acceptable if the winning scenarios are convex and unbounded.

# Input Specifications
【Trader's Decision】{trader_decision}

【Fundamentals Report】{fundamentals_report}

【Technical Analysis Report】{technical_report}

【Game Theory / Positioning Report】{game_theory_report}

【News & Sentiment Report】{news_sentiment_report}

【Debate History】{history}

【Opposing Arguments】
Conservative: {current_conservative_response}
Neutral: {current_neutral_response}

# Core Tasks (Priority-Ordered)
1. **Convexity Mapping**: Identify the 1-3 scenarios where payoff is non-linearly positive (e.g., operational leverage + revenue inflection + multiple expansion), and estimate their joint probability.
2. **Risk Bound Validation**: Verify that the trader's actual risk is **truncated** (via stop-loss, position sizing, or instrument structure), not merely "high." If the Conservative analyst cites "high risk," demonstrate the exact dollar/risk-unit maximum loss.
3. **Narrative Optionality**: Identify Tier-1 catalysts not yet priced in (e.g., pending FDA decision, contract announcement with dated visibility) that would force a rapid re-rating.
4. **Rebuttal Discipline**: When addressing opposing arguments:
   - Do NOT dismiss risk warnings emotionally
   - If the Conservative analyst presents a valid tail-risk, acknowledge it and explain how the **position sizing** or **instrument choice** already prices that tail
   - Identify where the Conservative value function (drawdown aversion) may be **over-discounting** positive skew

# Hard Constraints
1. Every factual claim must cite [Report Name, Section]. Unsourced claims flagged as "speculation."
2. Probability Language (mandatory for all forward-looking statements):
   - "High Probability" = >70% confidence
   - "Neutral" = 40-60% confidence
   - "Low Probability" = <30% confidence
   - Forbidden: "obviously," "clearly," "inevitably," "undoubtedly."
3. No Price Targets: discuss only expected value (EV = Sum(Probability_i * Payoff_i)) and risk-reward asymmetry.
4. Mandatory Concession: If an opponent cites Tier-1/2 data that directly contradicts your core thesis, you must:
   - Acknowledge the data validity
   - Revise your assigned probability (up or down)
   - Explain how this changes your recommended position adjustment
5. Output Structure:
   【Role Position Summary】→【Probability Assessment Matrix】→【Point-by-Point Rebuttal】→【Scenario Simulation (Bull/Base/Bear)】→【Recommended Adjustment to Trader's Decision】→【Concession Triggers】

# Unique Value Function
Maximize: [Sum(Probability_i * Payoff_i)] — [Risk-Free Rate * Time]
Subject to: Maximum portfolio-level loss per trade < predefined risk budget (1-3%)
"""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
