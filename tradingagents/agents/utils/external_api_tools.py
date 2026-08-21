"""统一外部API工具 — 每个Agent一个HTTP调用。"""

from langchain_core.tools import tool
from typing import Annotated
import requests

BASE_URL = "http://localhost:8000/api/external"


def _call(endpoint: str, ts_code: str, end_date: str = "") -> str:
    """通用HTTP GET调用，返回content字符串。"""
    url = f"{BASE_URL}/{endpoint}/{ts_code}"
    params = {}
    if end_date:
        params["end_date"] = end_date
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("content", "[API] content为空")
        return f"[API] HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return f"[API] 请求失败: {e}"


@tool
def tool_fundamental(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """基本面数据：财务指标、资产负债表、利润表、现金流量表、业绩预告/快报、机构预期、主营构成、行业PE等。"""
    return _call("fundamental", ts_code, end_date)


@tool
def tool_technical(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """技术面数据：日K线（前3个月）、分钟K线、资金流向、大盘拥挤度等。"""
    return _call("technical", ts_code, end_date)


@tool
def tool_game_theory(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """博弈面数据：返回内部人交易 + 高管增减持 + 股权质押明细 + 融资融券明细 + 资金流向等。"""
    return _call("game", ts_code, end_date)


@tool
def tool_risk(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """风险面数据：内部人交易、高管增减持、商誉余额、限售解禁等。"""
    return _call("risk", ts_code, end_date)


@tool
def tool_news_sentiment(
    ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
    end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
) -> str:
    """新闻舆情数据：个股研报（前2条）、个股新闻（前10条）、公告大全、互动易问答等。"""
    return _call("risk_sentiment", ts_code, end_date)


# @tool
# def tool_special_data(
#     ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"],
#     end_date: Annotated[str, "截止日期YYYY-MM-DD"] = "",
# ) -> str:
#     """特殊数据：市场环境综合评分(0-8)、趋势判断、主线行业TOP5、风险信号、仓位上限建议。"""
#     return _call("special_data", ts_code, end_date)
# NOTE: 外部 API 服务无 special_data 接口，此工具已停用（2026-08）

@tool
def position(ts_code: Annotated[str, "股票代码，格式如600519.SH或300394.SZ"])->str:
    """返回当前账户的可用资金总额，标的的持仓股数，最新收盘价，持仓市值(股数 × 最新收盘价)"""
    return _call("position",ts_code)