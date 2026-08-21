"""web_search_tool - optional networked search via DeepSeek's Responses API.

DeepSeek's Responses API (https://api.deepseek.com/responses) ships a
server-side web_search tool: the search is executed by the provider and
the result (text + citations) comes back in the response. We wrap that in a
plain LangChain tool so any agent that lists it among its tools can decide
autonomously whether to call it, e.g. to fetch supplementary material from
eastmoney.com for the instrument under analysis.

The tool is provider-agnostic at the agent level: it talks to DeepSeek with
its own API key (config web_search_api_key or the DEEPSEEK_API_KEY env
var), so it works even when the main analysis LLM is a local model.

Configuration (default_config.py):
- web_search_enabled: master switch (False by default)
- web_search_api_key / web_search_base_url / web_search_model
"""

from __future__ import annotations

import os
from typing import Annotated

from langchain_core.tools import tool
from openai import OpenAI


def deepseek_web_search(
    query: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_output: int = 3000,
) -> str:
    """Execute one server-side web search via the DeepSeek Responses API.

    Returns the search-result text (capped at max_output chars) with the
    citation URLs appended, or an error/empty marker as plain text so agents
    can react gracefully instead of crashing.
    """
    from tradingagents.dataflows.config import get_config

    cfg = get_config()
    key = api_key or cfg.get("web_search_api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return "[web_search] 未配置 DEEPSEEK_API_KEY（或 web_search_api_key），无法执行联网搜索"
    base = base_url or cfg.get("web_search_base_url") or "https://api.deepseek.com"
    mdl = model or cfg.get("web_search_model") or "deepseek-v4-flash"

    client = OpenAI(api_key=key, base_url=base)
    response = client.responses.create(
        model=mdl,
        input=query,
        tools=[{"type": "web_search"}],
        tool_choice={"type": "web_search"},
    )

    text = getattr(response, "output_text", None) or ""
    urls: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            for ann in getattr(content, "annotations", None) or []:
                url = getattr(ann, "url", None)
                if url and url not in urls:
                    urls.append(url)

    result = text.strip()[:max_output]
    if urls:
        result += "\n\n参考来源:\n" + "\n".join(f"- {u}" for u in urls[:10])
    return result or "[web_search] 未搜索到相关内容"


@tool
def web_search_tool(
    query: Annotated[
        str,
        "搜索查询。建议带上站点与股票代码以精准定位，例如："
        "site:eastmoney.com 600519.SH 财务报表；或 site:eastmoney.com 300394.SZ 最新公告",
    ],
    max_output: Annotated[int, "返回文本的最大长度（字符）"] = 3000,
) -> str:
    """联网搜索工具：通过 DeepSeek 内置联网搜索在东方财富（eastmoney.com）等网站检索补充信息，用于核实或补充股票数据、新闻与公告。"""
    return deepseek_web_search(query, max_output=max_output)
