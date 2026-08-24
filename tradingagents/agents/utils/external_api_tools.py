"""统一外部API工具 — 每个Agent一个HTTP调用。

外部数据服务（localhost:8000）接口约定：
- 数据接口返回结构化 JSON（schema_version / as_of / data / signals / warnings /
  source_status / errors），不再是旧版的 {content: ...} 字符串；
- /schema/{interface_name} 返回某接口的字段含义（path / summary / parameters /
  response_schema），供 LLM 在调用前了解字段语义；
- /market_environment 为市场宏观环境接口（全市场维度，无 ts_code）。
"""

import json
from typing import Annotated

import requests
from langchain_core.tools import tool

BASE_URL = "http://localhost:8000/api/external"


def once_per_run(tool):
    """Return a one-call-per-run wrapper for an external data tool.

    The ReAct guard (data_tool_done) only stops calls ACROSS LLM rounds; a
    model can still emit SEVERAL tool_calls for the same tool inside ONE
    message (e.g. trying different end_date), and the ToolNode executes them
    all. This wrapper answers every call after the first with a short marker
    instead of hitting the API again — so each data endpoint is fetched
    exactly once per run, no matter how many times the model requests it.
    """
    from langchain_core.tools import StructuredTool

    state = {"called": False}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return tool

    def _guarded(**kwargs):
        if state["called"]:
            return (
                "[API] 本次运行已取数：该接口只调用一次，"
                "请直接使用上下文中已有的工具返回撰写报告，不要重复请求。"
            )
        state["called"] = True
        # 直接调用原工具函数而不是 tool.invoke()：后者会在 LangChain 里再
        # 触发一组嵌套的 on_tool_start/on_tool_end 回调，导致一次调用在
        # 活动流中记录两条（原始字符串 + ToolMessage repr）。
        func = getattr(tool, "func", None)
        if callable(func):
            try:
                return func(**kwargs)
            except TypeError:  # 签名不匹配的兜底
                pass
        return tool.invoke(kwargs)

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=args_schema,
        func=_guarded,
    )


def _get(url: str) -> str:
    """通用HTTP GET调用，将响应体转为可读字符串返回。

    新格式为结构化 JSON，直接序列化返回（LLM 需要看完整的 schema_version /
    data / signals / warnings）；若响应仍为旧版 {content: ...} 则兼容返回
    content 字段。
    """
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                return resp.text
            if isinstance(data, dict) and data.get("content"):
                return str(data["content"])
            return json.dumps(data, ensure_ascii=False, indent=1)
        return f"[API] HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"[API] 请求失败: {e}"


def _call(endpoint: str, ts_code: str = "", end_date: str = "") -> str:
    """通用数据接口调用：GET {BASE_URL}/{endpoint}[/{ts_code}]?end_date=..."""
    url = f"{BASE_URL}/{endpoint}"
    if ts_code:
        url += f"/{ts_code}"
    params = {}
    if end_date:
        params["end_date"] = end_date
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return _get(url)


@tool
def tool_schema(
    interface_name: Annotated[str, "接口名，如 fundamental / technical / game / market_environment"],
) -> str:
    """查询外部数据接口的字段含义（path、summary、parameters、response_schema 字段描述）。

    字段语义会随版本变化，正式使用某个接口前应先用本工具查询该接口的字段含义，
    避免按旧字段名解读数据。
    """
    return _call("schema", interface_name)


@tool
def tool_fundamental(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """基本面数据（结构化JSON）：经济利润/护城河估值主锚（valuation）、
    财务质量、盈利能力、资产质量、资产负债表/利润表/现金流量表、业绩预告/快报、
    机构预期、行业PE等。返回字段含义请先调用 tool_schema('fundamental') 查询。"""
    return _call("fundamental", ts_code, end_date)


@tool
def tool_technical(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """技术面数据（结构化JSON）：methodology/数据新鲜度、最新收盘价、技术状态
    （趋势/RSI/KDJ/CCI/WR/布林/量价）与量价事件信号。返回字段含义请先
    调用 tool_schema('technical') 查询。"""
    return _call("technical", ts_code, end_date)


@tool
def tool_game_theory(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """博弈面数据（结构化JSON）：涨跌停统计、资金流向、龙虎榜、大宗交易、内部人
    交易/高管增减持、股权质押、融资融券、机构持仓等。返回字段含义请先调用
    tool_schema('game') 查询。"""
    return _call("game", ts_code, end_date)


@tool
def tool_market_environment(
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """市场宏观环境数据（全市场维度，无需 ts_code）：市场趋势结构（指数均线/
    ADX）、行业强弱分布、市场活跃度（涨跌停分布、主力资金流、沪深成交额）、
    隔夜美股行业表现等。返回字段含义请先调用 tool_schema('market_environment')
    查询。

    Args:
        end_date: 数据截止日期，YYYY-MM-DD；缺省为最新交易日。
    """
    return _call("market_environment", "", end_date)


@tool
def tool_news_sentiment(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """新闻舆情数据：个股研报、个股新闻、公告大全、互动易问答、机构调研等。"""
    return _call("risk_sentiment", ts_code, end_date)


@tool
def position(ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"])->str:
    """返回当前账户的可用资金总额，标的的持仓股数，最新收盘价，持仓市值(股数 × 最新收盘价)"""
    return _call("position", ts_code)
