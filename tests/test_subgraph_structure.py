"""Tests for the multi-subgraph decomposition of the trading pipeline.

The pipeline is split into five independently compiled stage subgraphs
(Analyst Team -> Research Debate -> Trader -> Risk Debate -> Portfolio
Manager) orchestrated by a thin parent graph. These tests verify:

1. the parent graph embeds exactly the five stage subgraphs,
2. each subgraph contains the expected internal nodes,
3. the full pipeline runs end-to-end with deterministic fake LLMs,
4. checkpoint resume still works when a run crashes inside a subgraph.
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

ALL_ANALYSTS = ["fundamentals", "technical", "game_theory", "news_sentiment"]


class StructuredProxy:
    """Returns a typed Pydantic instance for the requested schema."""

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


class FakeLLM:
    """Deterministic stand-in LLM: canned text, no tool calls, typed outputs."""

    def __init__(self, tag="fake"):
        self.tag = tag
        self.n_calls = 0

    def _respond(self, prompt=None):
        self.n_calls += 1
        return AIMessage(content=f"{self.tag} response #{self.n_calls}")

    def bind_tools(self, tools):
        return RunnableLambda(self._respond)

    def with_structured_output(self, schema, **kwargs):
        return StructuredProxy(schema)

    def invoke(self, prompt):
        return self._respond(prompt)


class CrashLLM(FakeLLM):
    """Fake LLM that crashes on the n-th call while armed (simulates a crash
    mid-run, e.g. a provider outage inside one of the stage subgraphs)."""

    def __init__(self, crash_after=8, tag="crash"):
        super().__init__(tag)
        self.crash_after = crash_after
        self.armed = True

    def _respond(self, prompt=None):
        self.n_calls += 1
        if self.armed and self.n_calls >= self.crash_after:
            raise RuntimeError("simulated mid-run crash")
        return AIMessage(content=f"{self.tag} response #{self.n_calls}")


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
    return config


def _build_graph(config, llm=None, analysts=None):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    llm = llm or FakeLLM()
    client = MagicMock()
    client.get_llm.return_value = llm
    with patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client):
        return TradingAgentsGraph(
            selected_analysts=ALL_ANALYSTS if analysts is None else analysts,
            debug=False,
            config=config,
        )


def _subgraph_nodes(workflow, name):
    compiled = workflow.nodes[name].runnable
    return sorted(n for n in compiled.get_graph().nodes if not n.startswith("__"))


STAGE_SUBGRAPHS = ["Analyst Team", "Research Debate", "Trader", "Risk Debate", "Portfolio Manager"]
TICKER_GUARDS = [f"TickerGuard-{s}" for s in STAGE_SUBGRAPHS]


@pytest.mark.unit
class TestParentGraphStructure:
    def test_parent_embeds_five_stage_subgraphs_with_ticker_guards(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        assert set(ta.workflow.nodes.keys()) == set(STAGE_SUBGRAPHS) | set(TICKER_GUARDS)
        for name in STAGE_SUBGRAPHS:
            assert hasattr(ta.workflow.nodes[name].runnable, "get_graph"), (
                f"{name} is not a compiled subgraph"
            )
        for name in TICKER_GUARDS:
            assert type(ta.workflow.nodes[name].runnable).__name__ != "CompiledStateGraph"

    def test_unknown_analyst_types_are_skipped(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path), analysts=["fundamentals", "bogus_type"])
        assert set(ta.workflow.nodes.keys()) == set(STAGE_SUBGRAPHS) | set(TICKER_GUARDS)
        analyst_nodes = _subgraph_nodes(ta.workflow, "Analyst Team")
        assert "Fundamentals Analyst" in analyst_nodes
        assert not any("Bogus" in n for n in analyst_nodes)

    def test_empty_analysts_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no analysts selected"):
            _build_graph(_make_config(tmp_path), analysts=[])


@pytest.mark.unit
class TestStageSubgraphInternals:
    def test_analyst_team_subgraph_nodes(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        nodes = _subgraph_nodes(ta.workflow, "Analyst Team")
        assert nodes == sorted([
            "Fundamentals Analyst", "Msg Clear Fundamentals", "tools_fundamentals",
            "Technical Analyst", "Msg Clear Technical", "tools_technical",
            "Game_Theory Analyst", "Msg Clear Game_Theory", "tools_game_theory",
            "News_Sentiment Analyst", "Msg Clear News_Sentiment", "tools_news_sentiment",
        ])

    def test_research_debate_subgraph_nodes(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        nodes = _subgraph_nodes(ta.workflow, "Research Debate")
        assert nodes == ["Bear Researcher", "Bull Researcher", "Research Manager"]

    def test_risk_debate_subgraph_nodes(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        nodes = _subgraph_nodes(ta.workflow, "Risk Debate")
        assert nodes == ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"]

    def test_trader_and_pm_subgraphs_have_single_node(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        assert _subgraph_nodes(ta.workflow, "Trader") == ["Trader"]
        assert _subgraph_nodes(ta.workflow, "Portfolio Manager") == ["Portfolio Manager"]


@pytest.mark.unit
class TestEndToEndPipeline:
    def test_full_pipeline_with_fake_llm(self, tmp_path):
        ta = _build_graph(_make_config(tmp_path))
        final_state, signal = ta.propagate("600519.SH", "2026-05-10")

        # All four analyst reports flow through the Analyst Team subgraph
        for key in (
            "fundamentals_report",
            "technical_report",
            "game_theory_report",
            "news_sentiment_report",
        ):
            assert final_state[key], f"{key} empty"

        # Debate budgets: 1 bull + 1 bear round; 1 aggressive + conservative + neutral
        assert final_state["investment_debate_state"]["count"] == 2
        assert final_state["risk_debate_state"]["count"] == 3

        # Structured outputs render through the Research Debate / Trader / PM stages
        assert final_state["investment_plan"].startswith("**Recommendation**")
        assert final_state["trader_investment_plan"].startswith("**Action**")
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

        # State log is persisted as before
        log_dir = tmp_path / "results" / "600519.SH" / "TradingAgentsStrategy_logs"
        assert list(log_dir.glob("full_states_log_2026-05-10.json"))


@pytest.mark.unit
class TestCheckpointResumeThroughSubgraphs:
    def test_crash_inside_subgraph_then_resume(self, tmp_path):
        """A crash inside a stage subgraph must be resumable from the checkpoint."""
        config = _make_config(tmp_path, checkpoint_enabled=True)
        llm = CrashLLM(crash_after=8)  # crash inside the Trader subgraph
        ta = _build_graph(config, llm=llm)

        with pytest.raises(RuntimeError, match="simulated mid-run crash"):
            ta.propagate("600519.SH", "2026-05-10")

        # The parent graph recorded a resumable checkpoint (super-step inside a subgraph)
        from tradingagents.graph.checkpointer import has_checkpoint
        assert has_checkpoint(config["data_cache_dir"], "600519.SH", "2026-05-10")

        # Resume: same graph instance, same thread id, same inputs
        llm.armed = False
        final_state, signal = ta.propagate("600519.SH", "2026-05-10")

        assert final_state["trader_investment_plan"].startswith("**Action**")
        assert final_state["final_trade_decision"].startswith("**Rating**")
        assert signal == "Buy"

        # Checkpoint is cleared after a successful run
        assert not has_checkpoint(config["data_cache_dir"], "600519.SH", "2026-05-10")
