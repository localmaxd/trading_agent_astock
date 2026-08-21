"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Analyst factual reports (structured claims + sources)
# ---------------------------------------------------------------------------


class AnalystClaim(BaseModel):
    """One factual claim from an analyst's report, with its provenance.

    Every quantitative statement an analyst makes must be traceable: the
    claim text, the value it carries, the tool that provided the original
    data, and the verbatim excerpt from that tool's output. The fact-checker
    node uses these to verify the report against freshly fetched data.
    """

    claim: str = Field(
        description="The conclusion or data point, one sentence.",
    )
    value: str = Field(
        description="The numerical value or qualitative conclusion as a string.",
    )
    source_tool: str = Field(
        description="Name of the tool that provided the original data, e.g. tool_fundamental / tool_technical.",
    )
    source_data: str = Field(
        description="Verbatim excerpt from the tool output backing this claim. Must be copied from the original tool result, never invented.",
    )


class AnalystFactualReport(BaseModel):
    """Structured analyst output: full markdown report + claim list.

    report_markdown keeps the existing prose report intact for downstream
    consumers (researchers, trader, saved files); claims is the
    machine-checkable list the fact-checker verifies.
    """

    summary: str = Field(
        description="One-paragraph summary of the overall assessment.",
    )
    claims: list[AnalystClaim] = Field(
        description="Every key data point / conclusion of the report with its source tool and raw source data.",
    )
    report_markdown: str = Field(
        description="The complete report in markdown, same shape as before (headings, tables, reasoning).",
    )


def render_factual_report(report: AnalystFactualReport) -> str:
    """Render an AnalystFactualReport back to plain markdown for downstream use."""
    return report.report_markdown


def claims_to_json(claims: list[AnalystClaim]) -> list[dict]:
    """Serialize claims to plain dicts for storage in the graph state."""
    return [c.model_dump() for c in claims]


# ---------------------------------------------------------------------------
# Fact verification (fact-checker node output)
# ---------------------------------------------------------------------------


class FactVerificationItem(BaseModel):
    """One verification result for a single claim in an analyst report."""

    claim: str = Field(description="The claim being verified (verbatim from the report).")
    verification_type: str = Field(
        description="Either 'fact' (cross-checked against tool data) or 'calculation' (re-computed).",
    )
    source_tool: str = Field(
        description="Tool whose freshly fetched data was used to verify this claim.",
    )
    reported_value: str = Field(
        description="Value stated in the analyst report.",
    )
    expected_value: str = Field(
        description="Value derived from the freshly fetched data / re-calculation.",
    )
    passed: bool = Field(
        description="True when reported and expected values agree within tolerance.",
    )
    difference: str = Field(
        default="",
        description="Quantified difference when the check failed, otherwise empty.",
    )
    failure_reason: str = Field(
        default="",
        description="Plain-language explanation of why the check failed (empty when passed).",
    )


class FactVerificationReport(BaseModel):
    """Structured output of the fact-checker node for one analyst."""

    items: list[FactVerificationItem] = Field(
        description="Verification results for every key claim in the report.",
    )
    overall_passed: bool = Field(
        description="True only when every item passed.",
    )
    feedback: str = Field(
        description="Concrete feedback for the analyst when overall_passed is False: which claims failed, what the expected values are, and what to redo. Empty when passed.",
    )


class VerificationSearchPlan(BaseModel):
    """Search plan produced by the fact-checker before the web-search rounds.

    The verifier LLM inspects the report and its claims, then decides which
    facts need confirmation on the public channel (eastmoney.com etc.).
    Every query in the list is executed by the fact-checker node in code; a
    second planning round may add follow-up queries based on the results
    already gathered.
    """

    queries: list[str] = Field(
        description=(
            "Search queries to execute (0-N). Each should target one fact of "
            "the report that needs public-channel confirmation, e.g. "
            "'site:eastmoney.com 600519.SH 业绩预告'. Facts already verifiable "
            "from the internal tool data (re-computable ratios, dual-source "
            "fund flow) do NOT need a search."
        ),
    )
    rationale: str = Field(
        default="",
        description="Why these searches are needed (one or two sentences).",
    )


def render_verification_report(verification: FactVerificationReport) -> str:
    """Render a FactVerificationReport to markdown (for logs / display)."""
    lines = [
        f"**Overall**: {'PASSED' if verification.overall_passed else 'FAILED'}",
        "",
        "| Claim | Type | Reported | Expected | Passed |",
        "|---|---|---|---|---|",
    ]
    for item in verification.items:
        lines.append(
            f"| {item.claim[:60]} | {item.verification_type} | "
            f"{item.reported_value[:40]} | {item.expected_value[:40]} | "
            f"{'PASS' if item.passed else 'FAIL'} |"
        )
    if verification.feedback:
        lines.extend(["", f"**Feedback**: {verification.feedback}"])
    return "\n".join(lines)

