"""Tests for web_search_tool (0820 feature: DeepSeek built-in web search).

The tool wraps DeepSeek's Responses API server-side web_search: it must
send the query with the web_search tool declared, and return the result
text plus citation URLs. All network access is mocked.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.utils.web_search_tool import deepseek_web_search


def _fake_response(text: str, urls: list[str]):
    items = []
    for u in urls:
        items.append(
            SimpleNamespace(
                content=[
                    SimpleNamespace(annotations=[SimpleNamespace(url=u)]),
                ]
            )
        )
    items.append(SimpleNamespace(content=[]))
    return SimpleNamespace(output_text=text, output=items)


@pytest.mark.unit
class TestDeepseekWebSearch:
    def _patch_client(self, fake_response):
        fake_client = MagicMock()
        fake_client.responses.create.return_value = fake_response
        patcher = patch("tradingagents.agents.utils.web_search_tool.OpenAI", return_value=fake_client)
        return patcher, fake_client

    def test_sends_query_with_web_search_tool(self, monkeypatch, tmp_path):
        patcher, fake_client = self._patch_client(_fake_response("东方财富财报显示营收增长20%", ["https://www.eastmoney.com/a/123"]))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with patcher:
            result = deepseek_web_search("site:eastmoney.com 600519.SH 财务报表")

        call_kwargs = fake_client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == "site:eastmoney.com 600519.SH 财务报表"
        assert call_kwargs["tools"] == [{"type": "web_search"}]
        assert call_kwargs["tool_choice"] == {"type": "web_search"}
        assert "东方财富财报显示营收增长20%" in result
        assert "https://www.eastmoney.com/a/123" in result

    def test_config_model_and_base_url_used(self, monkeypatch, tmp_path):
        from tradingagents.dataflows.config import get_config, set_config
        from tradingagents.default_config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG.copy()
        cfg["web_search_model"] = "deepseek-v4-pro"
        cfg["web_search_base_url"] = "https://api.deepseek.com"
        set_config(cfg)

        patcher, fake_client = self._patch_client(_fake_response("ok", []))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        try:
            with patcher:
                deepseek_web_search("query")
        finally:
            set_config(DEFAULT_CONFIG)

        call_kwargs = fake_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-v4-pro"

    def test_missing_api_key_returns_graceful_marker(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        patcher = patch("tradingagents.agents.utils.web_search_tool.OpenAI")
        with patcher as fake_cls:
            result = deepseek_web_search("any query")
        fake_cls.assert_not_called()
        assert "未配置" in result

    def test_no_results_returns_marker(self, monkeypatch):
        patcher, _ = self._patch_client(_fake_response("", []))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with patcher:
            result = deepseek_web_search("nothing here")
        assert result == "[web_search] 未搜索到相关内容"
