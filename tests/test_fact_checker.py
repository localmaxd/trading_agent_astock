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
    AnalystFactualReport,
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
    CROSS_VERIFY_TOOLS,
    create_fact_checker,
    make_verify_router,
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
        source_tool="tool_fundamental",
        source_data="ROIC 6.0%",
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
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    analyst_llm = analyst_llm or _FakeAnalystLLM()
    verify_llm = verify_llm or _FakeVerifyLLM([(True, "")])
    client = MagicMock()
    client.get_llm.side_effect = [verify_llm, analyst_llm]  # deep first, then quick

    patch_target = "tradingagents.graph.trading_graph.create_llm_client"
    patchers = [patch(patch_target, return_value=client)]
    if cross_tools is not None:
        patchers.append(patch(
            "tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS",
            cross_tools,
        ))
    for p in patchers:
        p.start()
    try:
        return TradingAgentsGraph(
            selected_analysts=["fundamentals", "technical", "game_theory", "news_sentiment"],
            debug=False,
            config=config,
        )
    finally:
        for p in patchers:
            p.stop()


@pytest.mark.unit
class TestCrossVerifyToolsDesign:
    def test_every_analyst_has_own_tool_and_a_real_cross_source(self):
        """Cross-checking is only meaningful where the same fact exists in
        another source: every analyst re-fetches its own tool AND has at
        least one other analyst's tool. Pure re-fetch is not cross-checking."""
        own_tool_names = {
            "fundamentals": "tool_fundamental",
            "technical": "tool_technical",
            "game_theory": "tool_game_theory",
        }
        for analyst_type, tools in CROSS_VERIFY_TOOLS.items():
            names = [t.name for t in tools]
            own = own_tool_names[analyst_type]
            assert own in names, f"{analyst_type} missing its own tool"
            assert any(n != own for n in names), (
                f"{analyst_type} has no cross-source tool"
            )

    def test_fundamentals_cross_checks_announcements_not_just_prices(self):
        """Financial ratios do not exist in tool_technical; the fundamentals
        cross-set must verify earnings forecasts against 公告大全 and use the
        price data only as an anchor."""
        names = [t.name for t in CROSS_VERIFY_TOOLS["fundamentals"]]
        assert "tool_news_sentiment" in names
        assert "tool_technical" in names

    def test_game_theory_cross_checks_insider_trades_via_risk_endpoint(self):
        """Insider/management trades appear in both game and risk endpoints;
        the risk endpoint is the independent cross-source for game_theory."""
        names = [t.name for t in CROSS_VERIFY_TOOLS["game_theory"]]
        assert "tool_risk" in names
        assert "tool_technical" in names  # fund-flow overlap

    def test_technical_cross_checks_fund_flow_and_trend(self):
        names = [t.name for t in CROSS_VERIFY_TOOLS["technical"]]
        assert "tool_game_theory" in names  # fund flow / limit prices
        # tool_special_data 已停用（外部 API 无此接口）——不再要求其在交叉集中
        # assert "tool_special_data" in names  # trend consistency

    def test_web_search_is_a_cross_source_for_every_verified_analyst(self):
        """Web search is an independent public-channel source: it must be
        part of every verified analyst's cross set (gated at runtime by
        web_search_enabled so an unconfigured key never pollutes the data)."""
        for analyst_type in ("fundamentals", "technical", "game_theory"):
            names = [t.name for t in CROSS_VERIFY_TOOLS[analyst_type]]
            assert "web_search_tool" in names, analyst_type


@pytest.mark.unit
class TestWebSearchGating:
    def _checker_with_web_search_fake(
        self,
        web_search_enabled,
        verify=None,
        ws=None,
    ):
        from tradingagents.dataflows.config import set_config
        from tradingagents.default_config import DEFAULT_CONFIG

        ws = ws or _FakeTool("web_search_tool", "东财公告：公司发布2026年一季报预告")
        other = _FakeTool("tool_technical", "close 100.5")
        verify = verify or _FakeVerifyLLM([(True, "")])
        checker = create_fact_checker("fundamentals", verify, max_rounds=2)

        cfg = DEFAULT_CONFIG.copy()
        cfg["web_search_enabled"] = web_search_enabled
        set_config(cfg)
        try:
            with patch(
                "tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS",
                {"fundamentals": [other, ws]},
            ), patch(
                "tradingagents.graph.fact_checker.web_search_tool",
                ws,
            ):
                out = checker({
                    "fundamentals_report": "# r",
                    "fundamentals_claims": [],
                    "company_of_interest": "600519.SH",
                    "trade_date": "2026-05-10",
                    "verification_state": {},
                })
        finally:
            set_config(DEFAULT_CONFIG)
        return ws, out

    def test_web_search_skipped_when_disabled(self):
        ws, out = self._checker_with_web_search_fake(web_search_enabled=False)
        assert ws.calls == []
        assert "web_search_tool" not in out["verification_state"]["fundamentals"].get("report_md", "")

    def test_web_search_planned_query_is_executed_when_enabled(self):
        verify = _FakeVerifyLLM(
            [(True, "")],
            search_plans=[
                ["site:eastmoney.com 600519.SH 业绩预告"],
                [],  # second planning round: nothing more needed
            ],
        )
        ws, _ = self._checker_with_web_search_fake(web_search_enabled=True, verify=verify)
        assert len(ws.calls) == 1
        query = ws.calls[0]["query"]
        assert "600519.SH" in query
        assert "eastmoney.com" in query


class _FlakyTool(_FakeTool):
    """Web-search fake that fails with a provider marker on the first call."""

    def __init__(self, name, data):
        super().__init__(name, data)
        self.fail_first = True

    def invoke(self, args: dict) -> str:
        self.calls.append(dict(args))
        if self.fail_first:
            self.fail_first = False
            return "[web_search] 未配置 DEEPSEEK_API_KEY（或 web_search_api_key），无法执行联网搜索"
        return self.data


@pytest.mark.unit
class TestMultiRoundWebSearch:
    def _run(self, verify, ws=None, cfg_extra=None):
        from tradingagents.dataflows.config import set_config
        from tradingagents.default_config import DEFAULT_CONFIG

        ws = ws or _FakeTool("web_search_tool", "东财公告内容")
        other = _FakeTool("tool_technical", "close 100.5")
        checker = create_fact_checker("fundamentals", verify, max_rounds=2)

        cfg = DEFAULT_CONFIG.copy()
        cfg["web_search_enabled"] = True
        cfg.update(cfg_extra or {})
        set_config(cfg)
        try:
            with patch(
                "tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS",
                {"fundamentals": [other, ws]},
            ), patch(
                "tradingagents.graph.fact_checker.web_search_tool",
                ws,
            ):
                checker({
                    "fundamentals_report": "# r",
                    "fundamentals_claims": [],
                    "company_of_interest": "600519.SH",
                    "trade_date": "2026-05-10",
                    "verification_state": {},
                })
        finally:
            set_config(DEFAULT_CONFIG)
        return ws

    def test_all_planned_queries_are_executed(self):
        verify = _FakeVerifyLLM(
            [(True, "")],
            search_plans=[
                ["site:eastmoney.com 600519.SH 业绩预告", "site:eastmoney.com 600519.SH 质押"],
                [],
            ],
        )
        ws = self._run(verify)
        assert [c["query"] for c in ws.calls] == [
            "site:eastmoney.com 600519.SH 业绩预告",
            "site:eastmoney.com 600519.SH 质押",
        ]

    def test_second_round_empty_stops_searching(self):
        verify = _FakeVerifyLLM(
            [(True, "")],
            search_plans=[
                ["site:eastmoney.com 600519.SH 业绩预告"],
                [],  # follow-up plan: enough results
            ],
        )
        ws = self._run(verify)
        assert len(ws.calls) == 1  # no second-round searches

    def test_provider_marker_is_retried_once(self):
        flaky = _FlakyTool("web_search_tool", "东财公告：业绩预告发布")
        verify = _FakeVerifyLLM(
            [(True, "")],
            search_plans=[
                ["site:eastmoney.com 600519.SH 业绩预告"],
                [],
            ],
        )
        ws = self._run(verify, ws=flaky)
        assert len(ws.calls) == 2  # first failed marker, retry succeeded
        assert ws.calls[1]["query"] == ws.calls[0]["query"]

    def test_max_queries_budget_is_capped(self):
        many = [f"site:eastmoney.com 600519.SH 主题{i}" for i in range(6)]
        verify = _FakeVerifyLLM(
            [(True, "")],
            search_plans=[many, []],
        )
        ws = self._run(verify, cfg_extra={"verify_search_max_queries": 4})
        assert len(ws.calls) == 4  # capped to the configured budget

    def test_planner_failure_falls_back_to_template_query(self):
        class _BrokenPlanLLM:
            """Planning rounds fail; the final verification pass still works."""

            def __init__(self):
                self.calls = 0

            def with_structured_output(self, schema, **kwargs):
                return self

            def invoke(self, prompt):
                self.calls += 1
                if self.calls <= 2:  # the two planning rounds fail
                    raise RuntimeError("planner backend down")
                return AIMessage(content="**Overall**: PASSED")

        verify = _BrokenPlanLLM()
        ws = self._run(verify)
        assert len(ws.calls) == 1  # template fallback fired once
        assert "600519.SH" in ws.calls[0]["query"]  # template still targeted
        assert "eastmoney.com" in ws.calls[0]["query"]


@pytest.mark.unit
class TestFactCheckerNode:
    def test_node_verifies_and_updates_state(self):
        tool = _FakeTool("tool_technical", "close 100.5")
        verify = _FakeVerifyLLM([(True, "")])
        checker = create_fact_checker("fundamentals", verify, max_rounds=2)
        with patch("tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS", {"fundamentals": [tool]}):
            out = checker({
                "fundamentals_report": "# report\nROIC 6.0%",
                "fundamentals_claims": [{"claim": "ROIC is 6.0%", "value": "6.0%", "source_tool": "tool_fundamental", "source_data": "ROIC 6.0%"}],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["attempts"] == 1
        assert vs["passed"] is True
        assert vs["items"] and vs["items"][0]["passed"] is True
        assert tool.calls == [{"ts_code": "600519.SH", "end_date": "2026-05-10"}]

    def test_node_records_failure_feedback(self):
        verify = _FakeVerifyLLM([(False, "recompute ROIC using EBIT/invested capital")])
        checker = create_fact_checker("fundamentals", verify, max_rounds=2)
        with patch("tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS", {"fundamentals": [_FakeTool("tool_technical")]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        vs = out["verification_state"]["fundamentals"]
        assert vs["passed"] is False
        assert "recompute ROIC" in vs["feedback"]

    def test_tool_failure_does_not_kill_stage(self):
        def boom(args):
            raise RuntimeError("backend down")
        bad = _FakeTool("tool_technical")
        bad.invoke = boom
        verify = _FakeVerifyLLM([(True, "")])
        checker = create_fact_checker("fundamentals", verify, max_rounds=2)
        with patch("tradingagents.graph.fact_checker.CROSS_VERIFY_TOOLS", {"fundamentals": [bad]}):
            out = checker({
                "fundamentals_report": "# r",
                "fundamentals_claims": [],
                "company_of_interest": "600519.SH",
                "trade_date": "2026-05-10",
                "verification_state": {},
            })
        assert out["verification_state"]["fundamentals"]["passed"] is True


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
        # Mirrors the real CROSS_VERIFY_TOOLS design: own tool (re-fetch) plus
        # cross-source tools chosen by actual data overlap.
        return {
            "fundamentals": [
                _FakeTool("tool_fundamental", "ROIC 6.0% 净利润 12.3亿"),
                _FakeTool("tool_news_sentiment", "公告大全：2026年一季报 净利润 12.3亿"),
                _FakeTool("tool_technical", "close 100.5"),
            ],
            "technical": [
                _FakeTool("tool_technical", "close 100.5\n资金净流入 1.2亿"),
                _FakeTool("tool_game_theory", "资金净流入 1.2亿\n涨停价 110.55"),
                # _FakeTool("tool_special_data", "趋势：上升\n市场环境评分 6"),  # 已停用
            ],
            "game_theory": [
                _FakeTool("tool_game_theory", "内部人净买入 500万股"),
                _FakeTool("tool_risk", "内部人净买入 500万股"),
                _FakeTool("tool_technical", "资金净流入 1.2亿"),
            ],
        }

    def test_failed_verification_retries_analyst_then_passes(self, tmp_path):
        """First verification FAILS -> analyst re-runs with feedback -> passes."""
        analyst = _FakeAnalystLLM()
        # fundamentals fails once then passes; other analysts pass immediately
        verify = _FakeVerifyLLM(by_analyst={
            "fundamentals": [(False, "ROIC 应为 8.1%，请用 EBIT/投入资本复算"), (True, "")],
        })
        ta = _build_graph(
            _make_config(tmp_path),
            analyst_llm=analyst,
            verify_llm=verify,
            cross_tools=self._cross_tools(),
        )
        final_state, signal = ta.propagate("600519.SH", "2026-05-10")

        vs = final_state["verification_state"]["fundamentals"]
        assert vs["attempts"] == 2, vs
        assert vs["passed"] is True
        # verify_llm doubles as the deep LLM (RM + PM also use it), so only
        # assert the verification path consumed its scripted verdicts.
        assert verify.n_calls >= 2

        # The analyst re-ran with the feedback injected into its prompt
        assert analyst.n_calls >= 2
        feedback_prompts = [p for p in analyst.seen_prompts if "ROIC 应为 8.1%" in p]
        assert feedback_prompts, "feedback was not injected into the analyst prompt"

        # The pipeline completed normally
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

    def test_failure_budget_exhausted_marks_unverified_and_continues(self, tmp_path):
        analyst = _FakeAnalystLLM()
        verify = _FakeVerifyLLM(by_analyst={
            "fundamentals": [(False, "still wrong"), (False, "still wrong")],
        })
        ta = _build_graph(
            _make_config(tmp_path),
            analyst_llm=analyst,
            verify_llm=verify,
            cross_tools=self._cross_tools(),
        )
        final_state, signal = ta.propagate("600519.SH", "2026-05-10")

        vs = final_state["verification_state"]["fundamentals"]
        assert vs["attempts"] == 2
        assert vs["passed"] is False

        # Pipeline is not blocked: later stages still ran
        assert final_state["trader_investment_plan"].startswith("**Action**")
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

    def test_verification_disabled_skips_checkers(self, tmp_path):
        config = _make_config(tmp_path, verify=False)
        ta = _build_graph(config)
        nodes = _subgraph_nodes(ta.workflow, "Analyst Team")
        assert not any(n.startswith("FactChecker") for n in nodes)


def _subgraph_nodes(workflow, name):
    compiled = workflow.nodes[name].runnable
    return sorted(n for n in compiled.get_graph().nodes if not n.startswith("__"))
