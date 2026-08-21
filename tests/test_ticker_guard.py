"""Tests for the per-stage ticker consistency guards.

Every stage subgraph in the parent graph is preceded by a TickerGuard that:
1. rejects malformed tickers (fail-fast),
2. rejects tickers that diverge from the run's anchor (input_ticker),
3. normalises the ticker (e.g. 600519 -> 600519.SH) before the stage runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.ticker_guard import (
    create_ticker_guard,
    is_valid_ticker,
    normalize_ticker,
)

ALL_ANALYSTS = ["fundamentals", "technical", "game_theory", "news_sentiment"]


@pytest.mark.unit
class TestNormalizeTicker:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("600519.SH", "600519.SH"),
            ("600519.sh", "600519.SH"),
            (" 600519.SH ", "600519.SH"),
            ("300394.SZ", "300394.SZ"),
            ("430047.BJ", "430047.BJ"),
            # bare A-share codes get completed (data-layer SH/SZ rule)
            ("600519", "600519.SH"),
            ("000001", "000001.SZ"),
            ("300394", "300394.SZ"),
            ("510300", "510300.SH"),  # 5-prefixed -> Shanghai
            # generic symbols are uppercased, suffix preserved
            ("cnc.to", "CNC.TO"),
            ("7203.T", "7203.T"),
            ("NVDA", "NVDA"),
            ("BRK.A", "BRK.A"),
            ("brk.a", "BRK.A"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert normalize_ticker(raw) == expected

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            None,
            123,
            "AAP L",
            "../etc/passwd",
            "a/b",
            "a\\b",
            "..",
            "600519.SH.extra",  # two dots are not a ticker
            "贵州茅台",
            "X" * 40,
        ],
    )
    def test_rejects_malformed(self, bad):
        assert not is_valid_ticker(bad)
        with pytest.raises(ValueError):
            normalize_ticker(bad)


@pytest.mark.unit
class TestTickerGuardNode:
    def _guard(self, stage="Test Stage"):
        return create_ticker_guard(stage)

    def test_passes_unchanged_when_consistent(self):
        node = self._guard()
        out = node({"company_of_interest": "600519.SH", "input_ticker": "600519.SH"})
        assert out == {}

    def test_completes_bare_a_share_code(self):
        node = self._guard()
        out = node({"company_of_interest": "600519", "input_ticker": "600519"})
        assert out == {"company_of_interest": "600519.SH"}

    def test_fixes_suffix_case(self):
        node = self._guard()
        out = node({"company_of_interest": "600519.sh", "input_ticker": "600519.SH"})
        assert out == {"company_of_interest": "600519.SH"}

    def test_rejects_malformed_ticker(self):
        node = self._guard("Research Debate")
        with pytest.raises(ValueError, match=r"TickerGuard:Research Debate.*not a valid ticker"):
            node({"company_of_interest": "../evil", "input_ticker": "600519.SH"})

    def test_rejects_ticker_diverging_from_anchor(self):
        """Simulates a previous stage rewriting the instrument."""
        node = self._guard("Research Debate")
        with pytest.raises(ValueError, match=r"TickerGuard:Research Debate.*inconsistency"):
            node({"company_of_interest": "000001.SZ", "input_ticker": "600519.SH"})

    def test_equivalent_forms_are_consistent(self):
        """A bare code and its completed form must not trip the guard."""
        node = self._guard()
        assert node({"company_of_interest": "600519", "input_ticker": "600519.SH"}) == {
            "company_of_interest": "600519.SH"
        }

    def test_anchor_missing_only_checks_format(self):
        node = self._guard()
        assert node({"company_of_interest": "600519"}) == {
            "company_of_interest": "600519.SH"
        }


class _StructuredProxy:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        name = self.schema.__name__
        if name == "PortfolioDecision":
            return self.schema(
                rating=PortfolioRating.BUY,
                executive_summary="Fake summary.",
                investment_thesis="Fake thesis.",
            )
        if name == "ResearchPlan":
            return self.schema(
                recommendation=PortfolioRating.OVERWEIGHT,
                rationale="Fake rationale.",
                strategic_actions="Fake actions.",
            )
        if name == "TraderProposal":
            return self.schema(action=TraderAction.BUY, reasoning="Fake reasoning.")
        raise NotImplementedError(name)


class _FakeLLM:
    def __init__(self, tag="fake"):
        self.tag = tag
        self.n_calls = 0

    def _respond(self, prompt=None):
        self.n_calls += 1
        return AIMessage(content=f"{self.tag} response #{self.n_calls}")

    def bind_tools(self, tools):
        return RunnableLambda(self._respond)

    def with_structured_output(self, schema, **kwargs):
        return _StructuredProxy(schema)

    def invoke(self, prompt):
        return self._respond(prompt)


def _make_config(tmp_path, checkpoint_enabled=False):
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "local"
    config["deep_think_llm"] = "fake-deep"
    config["quick_think_llm"] = "fake-quick"
    config["backend_url"] = "http://localhost:1/v1"
    config["results_dir"] = str(tmp_path / "results")
    config["data_cache_dir"] = str(tmp_path / "cache")
    config["memory_log_path"] = str(tmp_path / "memory.md")
    config["checkpoint_enabled"] = checkpoint_enabled
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["verify_enabled"] = False  # fact-checker covered in test_fact_checker.py
    return config


def _build_graph(config, llm=None):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    llm = llm or _FakeLLM()
    client = MagicMock()
    client.get_llm.return_value = llm
    with patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client):
        return TradingAgentsGraph(
            selected_analysts=ALL_ANALYSTS,
            debug=False,
            config=config,
        )


@pytest.mark.unit
class TestGuardPlacement:
    GUARDS = [
        "TickerGuard-Analyst Team",
        "TickerGuard-Research Debate",
        "TickerGuard-Trader",
        "TickerGuard-Risk Debate",
        "TickerGuard-Portfolio Manager",
    ]
    STAGES = ["Analyst Team", "Research Debate", "Trader", "Risk Debate", "Portfolio Manager"]

    def test_parent_graph_contains_guards_and_stages(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        nodes = set(ta.workflow.nodes.keys())
        assert nodes == set(self.GUARDS) | set(self.STAGES)

    def test_guards_are_functions_stages_are_subgraphs(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        for guard in self.GUARDS:
            runnable = ta.workflow.nodes[guard].runnable
            assert type(runnable).__name__ != "CompiledStateGraph", guard
        for stage in self.STAGES:
            runnable = ta.workflow.nodes[stage].runnable
            assert type(runnable).__name__ == "CompiledStateGraph", stage

    def test_every_stage_is_preceded_by_its_guard(self, tmp_path):
        """Edge-level check: the compiled parent graph routes
        START -> guard1 -> stage1 -> guard2 -> stage2 -> ... -> END."""
        ta = _build_graph(_make_config(tmp_path))
        edges = {(e.source, e.target) for e in ta.workflow.compile().get_graph().edges}
        assert ("__start__", "TickerGuard-Analyst Team") in edges
        for guard, stage in zip(self.GUARDS, self.STAGES):
            assert (guard, stage) in edges, f"guard {guard} does not precede {stage}"
        for i in range(len(self.STAGES) - 1):
            assert (self.STAGES[i], self.GUARDS[i + 1]) in edges, (
                f"{self.STAGES[i]} does not lead into {self.GUARDS[i + 1]}"
            )
        assert ("Portfolio Manager", "__end__") in edges


@pytest.mark.unit
class TestPollutionInterception:
    def test_pipeline_stops_if_a_stage_rewrites_the_ticker(self, tmp_path):
        """Replace the Analyst Team subgraph with a node that silently swaps
        the instrument; the next TickerGuard must abort the run."""
        from tradingagents.graph.conditional_logic import ConditionalLogic
        from tradingagents.graph.setup import GraphSetup

        llm = _FakeLLM()
        setup = GraphSetup(
            quick_thinking_llm=llm,
            deep_thinking_llm=llm,
            # The Analyst Team subgraph is overwritten below, so its ToolNode
            # never runs; a mock placeholder is enough to build the graph.
            tool_nodes={"fundamentals": MagicMock()},
            conditional_logic=ConditionalLogic(),
        )
        workflow = setup.setup_graph(["fundamentals"])

        def polluting_node(state):
            return {"company_of_interest": "000001.SZ"}  # swaps the instrument!

        spec = workflow.nodes["Analyst Team"]
        spec.runnable = polluting_node  # overwrite the compiled subgraph
        workflow.nodes["Analyst Team"] = spec

        graph = workflow.compile()
        init_state = Propagator().create_initial_state("600519.SH", "2026-05-10")

        with pytest.raises(ValueError, match=r"TickerGuard:Research Debate.*000001.SZ.*600519.SH"):
            graph.invoke(init_state)


@pytest.mark.unit
class TestEndToEndWithGuards:
    def test_bare_a_share_code_is_normalised_through_pipeline(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        final_state, signal = ta.propagate("600519", "2026-05-10")

        # The guard normalised the bare code before the Analyst Team stage
        assert final_state["company_of_interest"] == "600519.SH"
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

        # The persisted state log carries the canonical ticker too
        log_path = (
            tmp_path
            / "results"
            / "600519.SH"
            / "TradingAgentsStrategy_logs"
            / "full_states_log_2026-05-10.json"
        )
        assert log_path.exists()

    def test_canonical_ticker_still_runs(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        final_state, signal = ta.propagate("600519.SH", "2026-05-10")
        assert final_state["company_of_interest"] == "600519.SH"
        assert signal == "Buy"
