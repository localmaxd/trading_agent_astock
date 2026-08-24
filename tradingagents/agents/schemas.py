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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """5-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal.  Five actions cover the full
    position lifecycle:

    - Buy (建仓): open a new position — only possible from an empty account.
    - Add (加仓): increase an existing position.
    - Reduce (减仓): reduce an existing position (partial exit, e.g. taking
      profit / cutting risk) without fully exiting.
    - Sell (清仓): exit the position entirely.
    - Hold (持有): keep the current position unchanged.

    Position sizing and the nuanced Overweight / Underweight calls happen
    later at the Portfolio Manager.

    Position-dependent action rules:
    - EMPTY position (no holdings / no position info, treated as not holding
      anything): only Buy is allowed when entry conditions are met — Add /
      Reduce / Sell are impossible and Hold is meaningless for an empty
      account.
    - HOLDING a position: Add / Reduce / Sell / Hold are all allowed.
    """

    BUY = "Buy"
    ADD = "Add"
    REDUCE = "Reduce"
    SELL = "Sell"
    HOLD = "Hold"


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
        description=(
            "The transaction direction. Exactly one of Buy(建仓) / Add(加仓) / "
            "Reduce(减仓) / Sell(清仓) / Hold(持有). Must be consistent with the "
            "current position: if the account holds nothing (no position info "
            "defaults to empty), only Buy(建仓) is allowed and only when entry "
            "conditions are met — Add/Reduce/Sell are impossible and Hold is "
            "meaningless; when already holding a position, Add(加仓) / "
            "Reduce(减仓) / Sell(清仓) / Hold(持有) are all allowed."
        ),
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    quantity_shares: Optional[float] = Field(
        default=None,
        description=(
            "The CONCRETE number of shares to trade. MANDATORY for Add(加仓) and "
            "Reduce(减仓): compute it against the current holdings from the "
            "'position' tool (e.g. currently holding 500 shares -> Add 200 makes "
            "700, Reduce 200 leaves 300). Recommended for Buy(建仓)/Sell(清仓) too."
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


# 动作中文释义（渲染与展示用）
_TRADER_ACTION_LABELS = {
    TraderAction.BUY: "建仓买入",
    TraderAction.ADD: "加仓",
    TraderAction.REDUCE: "减仓",
    TraderAction.SELL: "清仓卖出",
    TraderAction.HOLD: "持有不动",
}


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/ADD/REDUCE/SELL/HOLD**``
    line is preserved for backward compatibility with the analyst stop-signal
    text and any external code that greps for it.
    """
    label = _TRADER_ACTION_LABELS.get(proposal.action, "")
    parts = [
        f"**Action**: {proposal.action.value}" + (f"（{label}）" if label else ""),
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.quantity_shares is not None:
        parts.extend(["", f"**Quantity**: {proposal.quantity_shares:g} shares"])
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


class ClaimEvidence(BaseModel):
    """A field-level reference into one immutable tool response.

    The LLM identifies *where* a value lives, but never supplies the value
    used by the verifier.  The fact-checker resolves ``json_path`` against the
    original ToolMessage retained in graph state.  ``alias`` is also the
    variable name used by a derived claim's arithmetic expression.
    """

    model_config = ConfigDict(extra="forbid")

    alias: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Short variable name, e.g. 'ebit' or 'invested'. For a calculated "
            "claim it must exactly match a variable referenced by formula."
        ),
    )
    source_tool: str = Field(
        description="Exact tool name that owns the JSON payload, e.g. tool_fundamental.",
    )
    json_path: str = Field(
        description=(
            "Exact path of the value in the original tool JSON. Use JSONPath "
            "such as '$.data.financial.ebit' or '$.data.rows[0].close'. "
            "For a list of values (consumed by sum/avg/median/max/min/count in "
            "formula or rule), map over the array: '$.data.rows[*].close'."
        ),
    )
    unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit of the referenced raw value when the JSON number itself is "
            "unitless, e.g. percent / CNY / shares / times."
        ),
    )
    period: Optional[str] = Field(
        default=None,
        description="Reporting period/as-of label for audit display, e.g. 2025FY or 2026-08-23.",
    )


class AnalystClaim(BaseModel):
    """One factual claim from an analyst's report, with its provenance.

    Every quantitative statement must point to exact fields in the immutable
    tool JSON. Derived numbers additionally carry an arithmetic formula whose
    variable names map to evidence aliases. The verifier resolves all values;
    the LLM never copies raw values into a trusted input field.
    """

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(
        description="The conclusion or data point, one sentence.",
    )
    value: str = Field(
        description=(
            "The scalar value shown in the report, including its display unit. "
            "Unsupported qualitative inferences must not be presented as code-verified facts."
        ),
    )
    evidence: list[ClaimEvidence] = Field(
        min_length=1,
        description=(
            "Field-level references resolved by code from the original tool JSON. "
            "A raw fact needs at least one reference; a calculated claim needs "
            "one alias/path reference for every formula variable."
        ),
    )
    formula: Optional[str] = Field(
        default=None,
        description=(
            "计算方式（仅计算得出的数据点需要）：算术表达式，如 'EBIT / invested'。"
            "变量名必须与 evidence.alias 一一对应；实际数值由代码按 JSON 路径读取。"
            "数组必须通过 sum/avg/mean/median/max/min/count 聚合；数组不能为空。"
            "无法给出计算方式的数据点不得列入 claims。formula 与 rule 只能填写一个。"
        ),
    )
    rule: Optional[str] = Field(
        default=None,
        description=(
            "规则/阈值判定（仅推理性判断需要）：布尔表达式，如 'roic > wacc'、"
            "'rsi > 70'、\"moat_strength == '高'\"。变量名与 evidence.alias 一一对应，"
            "实际值由代码按 JSON 路径读取后由代码求值；value 写结论（通过/未通过 或 真/假）。"
            "rule 必须引用 evidence 并返回真正的布尔值；不能使用常量或数值冒充布尔结果。"
            "代码无法求值、或求值结果与结论不符会判定失败。formula 与 rule 只能填写一个。"
        ),
    )

    @model_validator(mode="after")
    def validate_verification_shape(self) -> "AnalystClaim":
        """Reject ambiguous claims before they enter graph state."""
        if self.formula and self.formula.strip() and self.rule and self.rule.strip():
            raise ValueError("formula and rule are mutually exclusive")
        aliases = [item.alias for item in self.evidence]
        if len(aliases) != len(set(aliases)):
            raise ValueError("evidence aliases must be unique")
        return self


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
        description="Every key data point / conclusion with exact field-level evidence paths.",
    )
    report_markdown: str = Field(
        description="The complete report in markdown, same shape as before (headings, tables, reasoning).",
    )


class AnalystClaimSet(BaseModel):
    """Compact provenance extracted from an already-written analyst report."""

    summary: str = Field(
        description="Report summary, at most 100 Chinese characters.",
    )
    claims: list[AnalystClaim] = Field(
        description="At most 12 decision-relevant claims; do not repeat the report.",
    )


def render_factual_report(report: AnalystFactualReport) -> str:
    """Render an AnalystFactualReport back to plain markdown for downstream use."""
    return report.report_markdown


def claims_to_json(claims: list[AnalystClaim]) -> list[dict]:
    """Serialize claims to plain dicts for storage in the graph state."""
    from tradingagents.dataflows.config import get_config

    cfg = get_config()
    max_items = int(cfg.get("analyst_claim_max_items", 12))
    return [claim.model_dump() for claim in claims[:max_items]]


# ---------------------------------------------------------------------------
# Fact verification (fact-checker node output)
# ---------------------------------------------------------------------------


class ResolvedEvidence(BaseModel):
    """Evidence value resolved by code for audit/UI display."""

    alias: str
    source_tool: str
    json_path: str
    resolved_value: str
    unit: Optional[str] = None
    period: Optional[str] = None


class FactVerificationItem(BaseModel):
    """One verification result for a single claim in an analyst report."""

    claim: str = Field(description="The claim being verified (verbatim from the report).")
    verification_type: str = Field(
        description="Either 'fact' (cross-checked against tool data) or 'calculation' (re-computed).",
    )
    source_tool: str = Field(
        description="Tool(s) whose retained first-response data was used to verify this claim.",
    )
    reported_value: str = Field(
        description="Value stated in the analyst report.",
    )
    expected_value: str = Field(
        description="Value derived from the freshly fetched data / re-calculation.",
    )
    evidence: list[ResolvedEvidence] = Field(
        default_factory=list,
        description="Exact tool fields and values resolved by verification code.",
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
        "| Claim | Type | Resolved Evidence | Reported | Expected | Passed |",
        "|---|---|---|---|---|---|",
    ]
    for item in verification.items:
        evidence = "<br>".join(
            f"{entry.alias}←{entry.source_tool}:{entry.json_path}={entry.resolved_value}"
            for entry in item.evidence
        ) or "-"
        lines.append(
            f"| {item.claim[:60]} | {item.verification_type} | {evidence[:160]} | "
            f"{item.reported_value[:40]} | {item.expected_value[:40]} | "
            f"{'PASS' if item.passed else 'FAIL'} |"
        )
    if verification.feedback:
        lines.extend(["", f"**Feedback**: {verification.feedback}"])
    return "\n".join(lines)
