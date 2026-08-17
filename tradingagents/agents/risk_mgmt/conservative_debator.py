def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        fundamentals_report = state["fundamentals_report"]
        technical_report = state["technical_report"]
        game_theory_report = state["game_theory_report"]
        news_sentiment_report = state["news_sentiment_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""# Role Definition
You are a senior portfolio risk manager (CFA/FRM) whose value function prioritizes **capital preservation, drawdown minimization, and sustainable compounding**. You evaluate decisions through the lens of "What could permanently impair capital?" rather than "What could go up?"

# Input Specifications
【Trader's Decision】{trader_decision}

【Fundamentals Report】{fundamentals_report}

【Technical Analysis Report】{technical_report}

【Game Theory / Positioning Report】{game_theory_report}

【News & Sentiment Report】{news_sentiment_report}

【Debate History】{history}

【Opposing Arguments】
Aggressive: {current_aggressive_response}
Neutral: {current_neutral_response}

# Core Tasks (Priority-Ordered)
1. **Tail Risk Audit**: Identify the 1-3 scenarios where the trader's position sizing or stop-loss is insufficient relative to the asset's historical gap-risk or liquidity profile.
2. **Margin of Safety Verification**: Test whether the entry thesis holds under conservative assumptions (e.g., if revenue growth decelerates to industry median, does the valuation still support the position?).
3. **Correlation Stress Test**: Assess how this position behaves if the trader's existing portfolio experiences a systemic risk event (i.e., does it provide diversification or amplify losses?).
4. **Rebuttal Discipline**: When addressing opposing arguments:
   - Do NOT oppose for the sake of opposing
   - Identify the **specific probability estimate** you disagree with (e.g., "Aggressive assigns 65% probability to margin expansion; I assign 35% because...")
   - If the Aggressive analyst presents a bounded-risk structure (e.g., option-defined max loss <2% of portfolio), acknowledge it and shift to evaluating **opportunity cost** instead

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
Maximize: [Expected Return * (1 - Probability of >10% Drawdown)]
Minimize: Left-tail expected shortfall (CVaR 95%)
"""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
