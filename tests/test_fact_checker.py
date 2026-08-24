"""Tests for the fact-checker nodes (0820 feature: fact verification guard).

Covers:
1. node-level verification: re-fetches cross-source tools, produces a
   structured FactVerificationReport, updates verification_state,
2. router logic: next vs retry, bounded by max_verify_rounds,
3. end-to-end retry loop: failed verification sends the analyst back with
   feedback, a corrected report passes on the second attempt,
4. budget exhaustion: still-failing reports are marked unverified and the
   pipeline continues.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.schemas import (
    AnalystClaim,
    AnalystClaimSet,
    AnalystFactualReport,
    ClaimEvidence,
    FactVerificationItem,
    FactVerificationReport,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.fact_checker import (
    VERIFY_TOOLS,
    create_fact_checker,
    make_verify_router,
    verify_claims_in_code,
)


class _StructuredProxy:
    def __init__(self, schema, factory):
        self.schema = schema
        self.factory = factory

    def invoke(self, prompt):
        return self.factory(self.schema, prompt)


def _analyst_report(n_calls: int) -> AnalystFactualReport:
    """Canned structured analyst output (claims + markdown)."""
    claim = AnalystClaim(
        claim="ROIC is 6.0%",
        value="6.0%",
        evidence=[ClaimEvidence(
            alias="roic",
            source_tool="tool_fundamental",
            json_path="$.roic",
            unit="percent",
        )],
    )
    return AnalystFactualReport(
        summary=f"Fake summary {n_calls}",
        claims=[claim],
        report_markdown=f"# Fake report {n_calls}\nROIC 6.0%",
    )


def _verification_report(passed: bool, feedback: str) -> FactVerificationReport:
    item = FactVerificationItem(
        claim="ROIC is 6.0%",
        verification_type="calculation",
        source_tool="tool_fundamental",
        reported_value="6.0%",
        expected_value="6.0%",
        passed=passed,
        failure_reason="" if passed else "recomputed 8.1%, mismatch",
    )
    return FactVerificationReport(items=[item], overall_passed=passed, feedback=feedback)


def _make_schema_dispatching_proxy(schema, resolver):
    """Proxy that returns the correct Pydantic instance for the requested
    schema, falling back to a generic instance for schemas the test does not
    special-case (ResearchPlan / TraderProposal / PortfolioDecision used by
    the downstream stages). The prompt is forwarded so fakes can route by
    analyst (the parallel fact-checkers consume verdicts concurrently)."""

    def factory(s, prompt=None):
        name = s.__name__
        if name == "AnalystFactualReport":
            return resolver.analyst_report()
        if name == "AnalystClaimSet":
            report = resolver.analyst_report()
            return AnalystClaimSet(summary=report.summary, claims=report.claims)
        if name == "FactVerificationReport":
            return resolver.verification_report(prompt)
        if name == "VerificationSearchPlan":
            return resolver.search_plan()
        if name == "ResearchPlan":
            return ResearchPlan(
                recommendation=PortfolioRating.OVERWEIGHT,
                rationale="fake",
                strategic_actions="fake",
            )
        if name == "TraderProposal":
            return TraderProposal(action=TraderAction.BUY, reasoning="fake")
        if name == "PortfolioDecision":
            return PortfolioDecision(
                rating=PortfolioRating.BUY,
                executive_summary="fake",
                investment_thesis="fake",
            )
        return s.model_construct()

    return _StructuredProxy(schema, factory)


class _FakeAnalystLLM:
    """Stand-in analyst LLM. Records every rendered prompt so tests can
    assert the verify feedback was injected on retry."""

    def __init__(self, tag="analyst"):
        self.tag = tag
        self.n_calls = 0
        self.seen_prompts = []

    def analyst_report(self):
        return _analyst_report(self.n_calls)

    def verification_report(self):
        return _verification_report(True, "")

    def search_plan(self):
        from tradingagents.agents.schemas import VerificationSearchPlan
        return VerificationSearchPlan(queries=[])

    def _respond(self, prompt=None):
        self.n_calls += 1
        self.seen_prompts.append(str(prompt))
        return AIMessage(content=f"{self.tag} response #{self.n_calls}")

    def bind_tools(self, tools):
        return RunnableLambda(self._respond)

    def invoke(self, prompt):
        return self._respond(prompt)

    def with_structured_output(self, schema, **kwargs):
        return _make_schema_dispatching_proxy(schema, self)


def _analyst_report_with(value: str) -> AnalystFactualReport:
    """Canned analyst output with an arbitrary reported value (for retries)."""
    claim = AnalystClaim(
        claim=f"ROIC is {value}",
        value=value,
        evidence=[ClaimEvidence(
            alias="roic",
            source_tool="tool_fundamental",
            json_path="$.roic",
            unit="percent",
        )],
    )
    return AnalystFactualReport(
        summary="Fake summary",
        claims=[claim],
        report_markdown=f"# Fake report\nROIC {value}",
    )


class _RetryingAnalystLLM(_FakeAnalystLLM):
    """第一轮输出错误声明（代码校验失败），重试轮输出修正后的声明。"""

    def analyst_report(self):
        if self.n_calls <= 1:
            return _analyst_report_with("8.1%")
        return _analyst_report(self.n_calls)


class _AlwaysWrongAnalystLLM(_FakeAnalystLLM):
    """每次都输出无法通过代码校验的声明（预算耗尽场景）。"""

    def analyst_report(self):
        return _analyst_report_with("8.1%")


class _FakeVerifyLLM:
    """Verify LLM with scripted verdicts.

    The parallel fact-checkers call the verifier concurrently, so verdicts
    are routed PER ANALYST (detected from the prompt), not by global call
    order. by_analyst maps analyst_type -> list of (passed, feedback);
    verdicts is the fallback queue for any other caller.
    """

    def __init__(self, verdicts=None, search_plans=None, by_analyst=None):
        self.verdicts = list(verdicts or [(True, "")])
        self.n_calls = 0
        self.by_analyst = by_analyst or {}
        self._counts = {}
        # Scripted plans consumed per planning round; when None/empty the
        # planner returns an empty query list (no searches).
        self.search_plans = list(search_plans or [])

    def _analyst_from_prompt(self, prompt: str) -> str:
        text = str(prompt)
        for key in ("fundamentals", "technical", "game_theory"):
            if key in text:
                return key
        return ""

    def _next(self, key: str):
        self.n_calls += 1  # global call counter (parallel-safe: GIL)
        queue = self.by_analyst.get(key) or self.verdicts
        i = self._counts.get(key, 0)
        self._counts[key] = i + 1
        return queue[min(i, len(queue) - 1)]

    def analyst_report(self):
        return _analyst_report(0)

    def verification_report(self, prompt=None):
        passed, feedback = self._next(self._analyst_from_prompt(prompt))
        return _verification_report(passed, feedback)

    def search_plan(self):
        from tradingagents.agents.schemas import VerificationSearchPlan
        if self.search_plans:
            return VerificationSearchPlan(queries=self.search_plans.pop(0))
        return VerificationSearchPlan(queries=[])

    def with_structured_output(self, schema, **kwargs):
        return _make_schema_dispatching_proxy(schema, self)

    def invoke(self, prompt):
        passed, feedback = self._next()
        return AIMessage(
            content=f"**Overall**: {'PASSED' if passed else 'FAILED'}\n{feedback}"
        )


class _FakeTool:
    """Stand-in langchain tool that records calls and returns canned data."""

    def __init__(self, name, data="raw data 6.0%"):
        self.name = name
        self.data = data
        self.calls = []

    def invoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self.data


def _make_config(tmp_path, verify=True):
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "local"
    config["deep_think_llm"] = "fake-deep"
    config["quick_think_llm"] = "fake-quick"
    config["backend_url"] = "http://localhost:1/v1"
    config["results_dir"] = str(tmp_path / "results")
    config["data_cache_dir"] = str(tmp_path / "cache")
    config["memory_log_path"] = str(tmp_path / "memory.md")
    config["checkpoint_enabled"] = False
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["verify_enabled"] = verify
    config["max_verify_rounds"] = 2
    return config


def _build_graph(config, analyst_llm=None, verify_llm=None, cross_tools=None):
    """Build the graph keeping all patches ACTIVE until stopped (caller stops).

    The fact-checker reads VERIFY_TOOLS at runtime, so the patch must stay
    active while the pipeline propagates (otherwise the real HTTP tools run).
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    analyst_llm = analyst_llm or _FakeAnalystLLM()
    verify_llm = verify_llm or _FakeVerifyLLM([(True, "")])
    client = MagicMock()
    client.get_llm.side_effect = [verify_llm, analyst_llm]  # deep first, then quick

    patch_target = "tradingagents.graph.trading_graph.create_llm_client"
    patchers = [patch(patch_target, return_value=client)]
    if cross_tools is not None:
        patchers.append(patch(
            "tradingagents.graph.fact_checker.VERIFY_TOOLS",
            cross_tools,
        ))
    for p in patchers:
        p.start()
    ta = TradingAgentsGraph(
        selected_analysts=["fundamentals", "technical", "game_theory", "news_sentiment"],
        debug=False,
        config=config,
    )
    return ta, patchers


@pytest.mark.unit
class TestCrossVerifyToolsDesign:
    def test_each_analyst_self_checks_its_own_tool_only(self):
        """校验只检查"自己"：每个分析师只复拉并校验自身工具（自己检查自己的），
        不做跨源比对。"""
        own_tool_names = {
            "fundamentals": "tool_fundamental",
            "technical": "tool_technical",
            "game_theory": "tool_game_theory",
        }
        for analyst_type, tools in VERIFY_TOOLS.items():
            names = [t.name for t in tools]
            assert names == [own_tool_names[analyst_type]], (
                f"{analyst_type} should self-check exactly its own tool, got {names}"
            )

    def test_web_search_is_not_a_code_verification_source(self):
        """代码校验只对结构化 JSON 精确路径做确定性比对（字段取值 + 复算），
        web_search_tool 的自由文本检索结果无法代码级判定真伪，不进入校验集。"""
        for analyst_type in ("fundamentals", "technical", "game_theory"):
            names = [t.name for t in VERIFY_TOOLS[analyst_type]]
            assert "web_search_tool" not in names, analyst_type


@pytest.mark.unit
class TestFactCheckerNode:
    def test_claim_schema_rejects_removed_legacy_fields(self):
        """旧 source_data/inputs 不得再悄悄进入图状态。"""
        with pytest.raises(Exception):
            AnalystClaim(
                claim="ROIC is 6%",
                value="6%",
                evidence=[ClaimEvidence(
                    alias="roic",
                    source_tool="tool_fundamental",
                    json_path="$.roic",
                    unit="percent",
                )],
                source_data='"roic": 6.0',
            )

    def test_claim_schema_rejects_ambiguous_verification_shape(self):
        evidence = ClaimEvidence(
            alias="x", source_tool="tool_fundamental", json_path="$.x",
        )
        with pytest.raises(Exception, match="mutually exclusive"):
            AnalystClaim(
                claim="ambiguous", value="1", evidence=[evidence],
                formula="x", rule="x > 0",
            )
        with pytest.raises(Exception, match="aliases must be unique"):
            AnalystClaim(
                claim="duplicate", value="1", evidence=[evidence, evidence],
                formula="x",
            )

    def test_verification_reads_first_request_context_without_new_call(self):
        """校验直接读取分析师首次请求保留在上下文里的原始返回（ToolMessage），
        不发起新的取数请求（每个接口每次运行只被调用一次）。"""
        from langchain_core.messages import ToolMessage
        tool = _FakeTool("tool_fundamental", '{"roic": 6.0}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [{
                    "claim": "ROIC is 6.0%",
                    "value": "6.0%",
                    "evidence": [{
                        "alias": "roic", "source_tool": "tool_fundamental",
                        "json_path": "$.roic", "unit": "percent",
                    }],
                }],
                "messages_fundamentals": [
                    ToolMessage(content='{"roic": 6.0}', tool_call_id="call-1", name="tool_fundamental"),
                ],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is True
        assert tool.calls == [], "上下文已有首次请求数据，校验不应再次取数"

    def test_verification_reuses_context_across_retries(self):
        """重试轮同样读取首次请求上下文：验证两轮都不发起新请求。"""
        from langchain_core.messages import ToolMessage
        tool = _FakeTool("tool_fundamental", '{"roic": 6.0}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        state = {
            "fundamentals_report": "# r",
            "fundamentals_claims": [{
                "claim": "ROIC is 8.1%", "value": "8.1%",
                "evidence": [{
                    "alias": "roic", "source_tool": "tool_fundamental",
                    "json_path": "$.roic", "unit": "percent",
                }],
            }],
            "messages_fundamentals": [
                ToolMessage(content='{"roic": 6.0}', tool_call_id="call-1", name="tool_fundamental"),
            ],
            "company_of_interest": "600519.SH",
            "trade_date": "2026-05-10",
            "verification_state": {},
        }
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out1 = checker(state)
            # 第二轮（重试）：上下文仍保留首次数据，依然不取数
            state["verification_state"] = out1["verification_state"]
            out2 = checker(state)
        assert out1["verification_state"]["fundamentals"]["passed"] is False
        assert out2["verification_state"]["fundamentals"]["attempts"] == 2
        assert tool.calls == [], "两轮校验都不应发起新的取数请求"

    def test_node_verifies_and_updates_state(self):
        """字段路径解析值与报告值一致 → 代码校验通过（无 LLM 参与）。"""
        tool = _FakeTool("tool_fundamental", '{"roic": 6.0, "close": 100.5}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# report\nROIC 6.0%",
                "fundamentals_claims": [{
                    "claim": "ROIC is 6.0%",
                    "value": "6.0%",
                    "evidence": [{
                        "alias": "roic", "source_tool": "tool_fundamental",
                        "json_path": "$.roic", "unit": "percent",
                    }],
                }],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["attempts"] == 1
        assert vs["passed"] is True
        assert vs["items"] and vs["items"][0]["passed"] is True
        assert vs["items"][0]["verification_type"] == "fact"
        assert tool.calls == [{"ts_code": "600519.SH", "end_date": "2026-05-10"}]

    def test_exact_json_path_matches_value(self):
        """报告值只与 evidence 指定字段比较，不扫描 JSON 中的其他数字。"""
        tool = _FakeTool("tool_fundamental", '{"roic": 33.05}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [{
                    "claim": "ROIC is 33.05%",
                    "value": "33.05%",
                    "evidence": [{
                        "alias": "roic", "source_tool": "tool_fundamental",
                        "json_path": "$.roic", "unit": "percent",
                    }],
                }],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is True

    def test_node_records_failure_feedback(self):
        """字段路径解析值与报告值不匹配 → 代码校验失败并生成可操作反馈。"""
        tool = _FakeTool("tool_fundamental", '{"roic": 6.0}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [{
                    "claim": "ROIC is 8.1%",
                    "value": "8.1%",
                    "evidence": [{
                        "alias": "roic", "source_tool": "tool_fundamental",
                        "json_path": "$.roic", "unit": "percent",
                    }],
                }],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is False
        assert "解析出的值" in vs["feedback"]
        assert "ROIC is 8.1%" in vs["feedback"]
        assert vs["items"][0]["failure_reason"]

    def test_real_path_with_wrong_reported_value_does_not_pass(self):
        """证据路径真实存在也不能替错误报告值背书。"""
        result = verify_claims_in_code([{
            "claim": "ROIC is 99%",
            "value": "99%",
            "evidence": [{
                "alias": "roic", "source_tool": "tool_fundamental",
                "json_path": "$.roic", "unit": "percent",
            }],
        }], {"tool_fundamental": '{"roic": 6.0, "unrelated": 99.0}'})
        assert result.overall_passed is False
        assert result.items[0].expected_value == "6.0"

    def test_formula_ignores_spoofed_inputs_and_resolves_paths(self):
        """即使绕过 schema 注入 inputs，复算也只能使用 JSON 路径解析值。"""
        result = verify_claims_in_code([{
            "claim": "ROIC is 99%",
            "value": "99%",
            "formula": "ebit / invested",
            "inputs": {"ebit": 99, "invested": 100},
            "evidence": [
                {"alias": "ebit", "source_tool": "tool_fundamental", "json_path": "$.ebit"},
                {"alias": "invested", "source_tool": "tool_fundamental", "json_path": "$.invested"},
            ],
        }], {"tool_fundamental": '{"ebit": 3, "invested": 100}'})
        assert result.overall_passed is False
        assert result.items[0].expected_value == "3.00%"
        assert {e.alias: e.resolved_value for e in result.items[0].evidence} == {
            "ebit": "3", "invested": "100",
        }

    def test_report_quantity_without_claim_fails_coverage(self):
        """只提交一条正确 claim 不能掩盖报告里其他未校验数值。"""
        result = verify_claims_in_code([{
            "claim": "ROIC is 6%",
            "value": "6%",
            "evidence": [{
                "alias": "roic", "source_tool": "tool_fundamental",
                "json_path": "$.roic", "unit": "percent",
            }],
        }], {"tool_fundamental": '{"roic": 6.0, "net_profit": 1230000000}'},
            report="ROIC 为 6%，净利润为 12.3亿元。")
        assert result.overall_passed is False
        coverage = next(item for item in result.items if item.verification_type == "coverage")
        assert "12.3亿元" in coverage.reported_value

    def test_node_recomputes_formula_claims(self):
        """计算类声明：代码按 evidence 路径取值，再执行 formula 复算。"""
        tool = _FakeTool("tool_fundamental", '{"ebit": 330, "invested": 1000}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        claims = [{
            "claim": "ROIC is 33.0%",
            "value": "33.0%",
            "formula": "ebit / invested",
            "evidence": [
                {"alias": "ebit", "source_tool": "tool_fundamental", "json_path": "$.ebit"},
                {"alias": "invested", "source_tool": "tool_fundamental", "json_path": "$.invested"},
            ],
        }, {
            "claim": "ROIC is 31.0%",
            "value": "31.0%",
            "formula": "ebit / invested",
            "evidence": [
                {"alias": "ebit", "source_tool": "tool_fundamental", "json_path": "$.ebit"},
                {"alias": "invested", "source_tool": "tool_fundamental", "json_path": "$.invested"},
            ],
        }]
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": claims,
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is False
        items = vs["items"]
        assert items[0]["passed"] is True
        assert items[0]["expected_value"] == "33.00%"
        assert items[1]["passed"] is False
        assert "复算得 33.00%" in items[1]["failure_reason"]
        assert "31.0%" in items[1]["failure_reason"]

    def test_aggregate_formula_over_array_evidence(self):
        """数组聚合：sum/median 参与公式，代码从 json_path[*] 取值计算。"""
        tool = _FakeTool("tool_technical", '{"rows": [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]}')
        checker = create_fact_checker("technical", max_rounds=2)
        claims = [
            {
                "claim": "四日合计为 10",
                "value": "10",
                "formula": "sum(vs)",
                "evidence": [{"alias": "vs", "source_tool": "tool_technical", "json_path": "$.rows[*].v"}],
            },
            {
                "claim": "中位数为 2.5",
                "value": "2.5",
                "formula": "median(vs)",
                "evidence": [{"alias": "vs", "source_tool": "tool_technical", "json_path": "$.rows[*].v"}],
            },
            {
                "claim": "合计大于 5",
                "value": "通过",
                "rule": "sum(vs) > 5",
                "evidence": [{"alias": "vs", "source_tool": "tool_technical", "json_path": "$.rows[*].v"}],
            },
            {
                "claim": "合计写错",
                "value": "9",
                "formula": "sum(vs)",
                "evidence": [{"alias": "vs", "source_tool": "tool_technical", "json_path": "$.rows[*].v"}],
            },
        ]
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"technical": [tool]}):
            out = checker({
                "technical_report": "# r",
                "technical_claims": claims,
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["technical"]
        assert vs["passed"] is False
        items = vs["items"]
        assert items[0]["verification_type"] == "calculation" and items[0]["passed"] is True
        assert items[0]["expected_value"] == "10"
        assert items[1]["passed"] is True and items[1]["expected_value"] == "2.5"
        assert items[2]["verification_type"] == "rule" and items[2]["passed"] is True
        assert items[3]["passed"] is False
        assert "复算得 10" in items[3]["failure_reason"]

    def test_rule_inference_evaluated_in_code(self):
        """推理性/阈值判断：代码按 evidence 路径取值并求值布尔规则，再核对结论。"""
        tool = _FakeTool("tool_fundamental", '{"roic": 33.05, "wacc": 8.5, "moat_strength": "高"}')
        checker = create_fact_checker("fundamentals", max_rounds=2)
        claims = [
            {
                "claim": "ROIC 高于 WACC，通过经济利润检验",
                "value": "通过",
                "rule": "roic > wacc",
                "evidence": [
                    {"alias": "roic", "source_tool": "tool_fundamental", "json_path": "$.roic", "unit": "percent"},
                    {"alias": "wacc", "source_tool": "tool_fundamental", "json_path": "$.wacc", "unit": "percent"},
                ],
            },
            {
                "claim": "ROIC 高于 WACC（结论写反）",
                "value": "未通过",
                "rule": "roic > wacc",
                "evidence": [
                    {"alias": "roic", "source_tool": "tool_fundamental", "json_path": "$.roic", "unit": "percent"},
                    {"alias": "wacc", "source_tool": "tool_fundamental", "json_path": "$.wacc", "unit": "percent"},
                ],
            },
            {
                "claim": "护城河强度为高",
                "value": "通过",
                "rule": "moat_strength == '高'",
                "evidence": [
                    {"alias": "moat_strength", "source_tool": "tool_fundamental", "json_path": "$.moat_strength"},
                ],
            },
        ]
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": claims,
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is False
        items = vs["items"]
        assert items[0]["verification_type"] == "rule" and items[0]["passed"] is True
        assert items[1]["passed"] is False
        assert "求值为 通过" in items[1]["failure_reason"]
        assert items[2]["verification_type"] == "rule" and items[2]["passed"] is True

    def test_array_or_object_cannot_pass_as_scalar_fact(self):
        """通配路径得到的集合不能再通过字符串中的首个数字冒充标量事实。"""
        raw = {"tool_technical": '{"rows": [{"v": 1}, {"v": 2}], "obj": {"v": 1}}'}
        claims = [
            {
                "claim": "列表错误冒充 1",
                "value": "1",
                "evidence": [{
                    "alias": "values", "source_tool": "tool_technical",
                    "json_path": "$.rows[*].v",
                }],
            },
            {
                "claim": "对象错误冒充 1",
                "value": "1",
                "evidence": [{
                    "alias": "obj", "source_tool": "tool_technical",
                    "json_path": "$.obj",
                }],
            },
        ]
        result = verify_claims_in_code(claims, raw)
        assert result.overall_passed is False
        assert all(item.passed is False for item in result.items)
        assert all("普通 fact 必须引用单个标量" in item.failure_reason for item in result.items)

    def test_rule_requires_boolean_result_and_referenced_evidence(self):
        """数值/字符串 truthiness 和无证据常量规则都不能充当推理结论。"""
        raw = {"tool_fundamental": '{"x": 7}'}
        claims = [
            {
                "claim": "数值不能冒充布尔值",
                "value": "通过",
                "rule": "x",
                "evidence": [{"alias": "x", "source_tool": "tool_fundamental", "json_path": "$.x"}],
            },
            {
                "claim": "常量比较不能绕开证据",
                "value": "通过",
                "rule": "1 < 2",
                "evidence": [{"alias": "x", "source_tool": "tool_fundamental", "json_path": "$.x"}],
            },
        ]
        result = verify_claims_in_code(claims, raw)
        assert result.overall_passed is False
        assert result.items[0].passed is False
        assert "布尔值" in result.items[0].failure_reason
        assert result.items[1].passed is False
        assert "不能使用常量规则" in result.items[1].failure_reason

    def test_claim_shape_rejects_conflicts_duplicates_and_unused_evidence(self):
        raw = {"tool_fundamental": '{"x": 2, "y": 3}'}
        evidence_x = {"alias": "x", "source_tool": "tool_fundamental", "json_path": "$.x"}
        claims = [
            {
                "claim": "formula/rule 冲突", "value": "2",
                "formula": "x", "rule": "x > 0", "evidence": [evidence_x],
            },
            {
                "claim": "重复 alias", "value": "2", "formula": "x",
                "evidence": [evidence_x, {
                    "alias": "x", "source_tool": "tool_fundamental", "json_path": "$.y",
                }],
            },
            {
                "claim": "未使用 evidence", "value": "2", "formula": "x",
                "evidence": [evidence_x, {
                    "alias": "y", "source_tool": "tool_fundamental", "json_path": "$.y",
                }],
            },
        ]
        result = verify_claims_in_code(claims, raw)
        assert [item.passed for item in result.items] == [False, False, False]
        assert "不能同时填写" in result.items[0].failure_reason
        assert "重复" in result.items[1].failure_reason
        assert "未被 formula 使用" in result.items[2].failure_reason

    def test_aggregate_boundaries_reject_empty_nested_null_and_non_finite_data(self):
        raw = {"tool_technical": json.dumps({
            "empty": [],
            "groups": [{"rows": [{"v": 1}]}, {"rows": [{"v": 2}]}],
            "rows": [{"v": 1}, {"v": None}],
            "non_finite": float("nan"),
        })}
        claims = [
            {
                "claim": "空数组", "value": "0", "formula": "sum(vs)",
                "evidence": [{"alias": "vs", "source_tool": "tool_technical", "json_path": "$.empty[*].v"}],
            },
            {
                "claim": "嵌套数组", "value": "3", "formula": "sum(vs)",
                "evidence": [{
                    "alias": "vs", "source_tool": "tool_technical",
                    "json_path": "$.groups[*].rows[*].v",
                }],
            },
            {
                "claim": "数组含 null", "value": "1", "formula": "sum(vs)",
                "evidence": [{"alias": "vs", "source_tool": "tool_technical", "json_path": "$.rows[*].v"}],
            },
            {
                "claim": "非有限数", "value": "0", "formula": "x * 2",
                "evidence": [{
                    "alias": "x", "source_tool": "tool_technical", "json_path": "$.non_finite",
                }],
            },
        ]
        result = verify_claims_in_code(claims, raw)
        assert [item.passed for item in result.items] == [False, False, False, False]
        assert "空数组" in result.items[0].failure_reason
        assert "嵌套数组/对象" in result.items[1].failure_reason
        assert "不是有限数值" in result.items[2].failure_reason
        assert "不是有限数值" in result.items[3].failure_reason

    def test_bounded_interpreter_handles_cagr_but_rejects_explosive_power(self):
        raw = {"tool_fundamental": '{"start": 100, "end": 121, "years": 2, "x": 10}'}
        result = verify_claims_in_code([
            {
                "claim": "两年 CAGR", "value": "10%",
                "formula": "(end / start) ** (1 / years) - 1",
                "evidence": [
                    {"alias": "start", "source_tool": "tool_fundamental", "json_path": "$.start"},
                    {"alias": "end", "source_tool": "tool_fundamental", "json_path": "$.end"},
                    {"alias": "years", "source_tool": "tool_fundamental", "json_path": "$.years"},
                ],
            },
            {
                "claim": "爆炸乘方", "value": "1",
                "formula": "10 ** (10 ** x)",
                "evidence": [{"alias": "x", "source_tool": "tool_fundamental", "json_path": "$.x"}],
            },
        ], raw)
        assert result.items[0].passed is True
        assert result.items[0].expected_value == "10.00%"
        assert result.items[1].passed is False
        assert "无法从原始工具数据复算" in result.items[1].failure_reason

    def test_empty_claims_fails_verification(self):
        """无结构化声明时无法代码校验 → 失败并反馈分析师重新输出。"""
        checker = create_fact_checker("fundamentals", max_rounds=2)
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": []}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is False
        assert "claims 为空" in vs["feedback"]

    def test_tool_failure_does_not_kill_stage(self):
        def boom(args):
            raise RuntimeError("backend down")
        bad = _FakeTool("tool_technical")
        bad.invoke = boom
        checker = create_fact_checker("fundamentals", max_rounds=2)
        with patch("tradingagents.graph.fact_checker.VERIFY_TOOLS", {"fundamentals": [bad]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        # 取数失败不导致节点崩溃：校验照常产出（claims 为空 → 未通过）
        assert "attempts" in vs and vs["attempts"] == 1
        assert vs["passed"] is False


@pytest.mark.unit
class TestVerifyRouter:
    def test_passed_goes_next(self):
        router = make_verify_router("fundamentals", max_rounds=2)
        assert router({"verification_state": {"fundamentals": {"passed": True, "attempts": 1}}}) == "next"

    def test_failed_within_budget_retries(self):
        router = make_verify_router("fundamentals", max_rounds=2)
        assert router({"verification_state": {"fundamentals": {"passed": False, "attempts": 1}}}) == "retry"

    def test_failed_budget_exhausted_goes_next(self):
        router = make_verify_router("fundamentals", max_rounds=2)
        assert router({"verification_state": {"fundamentals": {"passed": False, "attempts": 2}}}) == "next"


@pytest.mark.unit
class TestEndToEndVerification:
    def _cross_tools(self):
        # 自查：每个分析师只校验自身工具（自己检查自己的，无跨源比对）。
        return {
            "fundamentals": [
                _FakeTool("tool_fundamental", '{"roic": 6.0, "net_profit": 1230000000}'),
            ],
            "technical": [
                _FakeTool("tool_technical", '{"close": 100.5, "net_inflow": 120000000}'),
            ],
            "game_theory": [
                _FakeTool("tool_game_theory", '{"insider_net_buy": 5000000}'),
            ],
        }

    def test_failed_verification_retries_analyst_then_passes(self, tmp_path):
        """代码校验第一次 FAILS -> 分析师收到反馈重做（不重新取数）-> 第二次通过。"""
        analyst = _RetryingAnalystLLM()
        # deep LLM 只用于 RM/PM 的结构化输出代理（校验环节不再调用 LLM）
        verify = _FakeVerifyLLM([(True, "")])
        ta, patchers = _build_graph(
            _make_config(tmp_path),
            analyst_llm=analyst,
            verify_llm=verify,
            cross_tools=self._cross_tools(),
        )
        try:
            final_state, signal = ta.propagate("600519.SH", "2026-05-10")
        finally:
            for p in patchers:
                p.stop()

        vs = final_state["verification_state"]["fundamentals"]
        assert vs["attempts"] == 2, vs
        assert vs["passed"] is True

        # 分析师重跑并收到代码校验反馈（错误原因注入其 prompt）
        assert analyst.n_calls >= 2
        feedback_prompts = [p for p in analyst.seen_prompts if "解析出的值" in p]
        assert feedback_prompts, "code-check feedback was not injected into the analyst prompt"

        # The pipeline completed normally
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

    def test_failure_budget_exhausted_marks_unverified_and_continues(self, tmp_path):
        analyst = _AlwaysWrongAnalystLLM()
        verify = _FakeVerifyLLM([(True, "")])
        ta, patchers = _build_graph(
            _make_config(tmp_path),
            analyst_llm=analyst,
            verify_llm=verify,
            cross_tools=self._cross_tools(),
        )
        try:
            final_state, signal = ta.propagate("600519.SH", "2026-05-10")
        finally:
            for p in patchers:
                p.stop()

        vs = final_state["verification_state"]["fundamentals"]
        assert vs["attempts"] == 2
        assert vs["passed"] is False
        assert "解析出的值" in vs["feedback"]

        # Pipeline is not blocked: later stages still ran
        assert final_state["trader_investment_plan"].startswith("**Action**")
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

    def test_verification_disabled_skips_checkers(self, tmp_path):
        config = _make_config(tmp_path, verify=False)
        ta, patchers = _build_graph(config)
        try:
            team = ta.workflow.nodes["Analyst Team"].runnable
            for analyst_name, node in team.get_graph().nodes.items():
                if analyst_name.startswith("__"):
                    continue
                nodes = node.data.get_graph().nodes
                assert not any(n.startswith("FactChecker") for n in nodes)
        finally:
            for p in patchers:
                p.stop()


def _subgraph_nodes(workflow, name):
    compiled = workflow.nodes[name].runnable
    return sorted(n for n in compiled.get_graph().nodes if not n.startswith("__"))
