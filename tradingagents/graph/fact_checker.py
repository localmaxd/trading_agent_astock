"""Fact-checker nodes for the analyst stage.

After fundamentals / technical / game_theory produce their report, a
fact-checker node verifies the claims before the pipeline moves on:

1. It re-fetches data in code: the analyst's own tool (reproducibility) plus
   at least one other analyst's tool (cross-source check).
2. It hands the report, the structured claims, and the freshly fetched raw
   data to a verifier LLM which emits a FactVerificationReport: every claim
   is either cross-checked against the raw data (fact) or re-computed
   (calculation).
3. On failure the feedback is stored in verification_state; the subgraph
   router sends the analyst back to redo its material (bounded by
   max_verify_rounds). Beyond the budget the report is marked unverified and
   the pipeline continues, so a weak data point can never block the run
   forever.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

from tradingagents.agents.schemas import (
    FactVerificationReport,
    render_verification_report,
)
from tradingagents.agents.utils.external_api_tools import (
    tool_fundamental,
    tool_game_theory,
    tool_news_sentiment,
    tool_risk,
    # tool_special_data,  # 外部 API 无此接口，已停用（2026-08）
    tool_technical,
)
from tradingagents.agents.utils.web_search_tool import web_search_tool
from tradingagents.dataflows.config import get_config
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-source tool sets.
#
# Design principle: cross-checking is only meaningful where the SAME fact
# appears in more than one data source. Each analyst therefore gets:
#
#   1. its own tool  -> re-fetch (reproducibility): the verifier re-computes
#      ratios/indicators from freshly fetched data and catches claims whose
#      quoted "source_data" does not match what the tool actually returns
#      (i.e. LLM-invented or stale excerpts);
#   2. other analysts' tools -> genuine cross-source checks, chosen by the
#      actual data overlap between endpoints (see comments below), NOT by
#      analyst identity.
#
# Data overlap map (from the akshare data layer):
#   - tool_news_sentiment returns 公告大全 (announcements) incl. 业绩预告/
#     财报公告 -> cross-checks fundamentals' earnings-forecast claims.
#   - tool_technical returns 日K (latest close) -> anchor for any price /
#     market-cap statements in the fundamentals report.
#   - tool_game_theory returns 个股资金流向 + 涨跌停价格 -> cross-checks
#     technical's fund-flow / price-move claims.
#   - tool_special_data returns 趋势判断/市场环境 -> consistency check for
#     technical's trend conclusions.  [已停用：外部 API 无 special_data 接口]
#   - tool_risk returns 内部人交易/高管增减持 through an independent endpoint
#     -> the strongest cross-source overlap with game_theory.
# ---------------------------------------------------------------------------
CROSS_VERIFY_TOOLS: Dict[str, List[Any]] = {
    # Financial ratios exist ONLY in tool_fundamental -> re-fetch + recompute.
    # Cross-checks: earnings-forecast existence vs 公告大全; price/market-cap
    # anchors vs 日K latest close; public-channel existence vs web search.
    "fundamentals": [
        tool_fundamental,        # 复拉：财务指标/三表，供复算 ROIC/毛利率/ROE 等
        tool_news_sentiment,     # 交叉：公告大全 → 业绩预告/财报公告的真实性与数值
        tool_technical,          # 交叉：日K 最新收盘价 → 报告中的价格/市值锚点
        web_search_tool,         # 交叉：公开渠道验证业绩预告/重大财务事件的真实发布
    ],
    # OHLCV lives only in tool_technical -> re-fetch + recompute MA/MACD/RSI.
    # Cross-checks: fund flow & limit prices vs game_theory endpoint;
    # price-affecting events vs web search.
    "technical": [
        tool_technical,          # 复拉：日K/指标，供复算均线、MACD、RSI、涨跌幅
        tool_game_theory,        # 交叉：个股资金流向 + 涨跌停价格（与 K 线/资金流一致）
        # tool_special_data,     # 交叉：趋势判断/市场环境（已停用：外部 API 无此接口）
        web_search_tool,         # 交叉：停牌/除权除息/重大新闻等影响价格的事件
    ],
    # Insider/management trades exist in BOTH tool_game_theory and tool_risk
    # (independent endpoints) -> the cleanest cross-source check in the set.
    "game_theory": [
        tool_game_theory,        # 复拉：内部人/质押/两融/龙虎榜，供复算
        tool_risk,               # 交叉：内部人交易 + 高管增减持（独立端点同一事实）
        tool_technical,          # 交叉：个股资金流向（两端点均返回资金流向）
        web_search_tool,         # 交叉：公开渠道验证减持/质押/龙虎榜公告
    ],
}

# Web-search query templates per analyst (the search runs on eastmoney.com
# via DeepSeek's built-in web search; the ts_code is substituted at call time).
_WEB_SEARCH_QUERIES: Dict[str, str] = {
    "fundamentals": "site:eastmoney.com {ts_code} 财务报表 业绩预告 公告",
    "technical": "site:eastmoney.com {ts_code} 股价 停牌 除权 重大公告",
    "game_theory": "site:eastmoney.com {ts_code} 减持 质押 龙虎榜 公告",
}

VERIFY_ANALYSTS = ("fundamentals", "technical", "game_theory")

_REPORT_KEYS = {
    "fundamentals": "fundamentals_report",
    "technical": "technical_report",
    "game_theory": "game_theory_report",
}

_CLAIM_KEYS = {
    "fundamentals": "fundamentals_claims",
    "technical": "technical_claims",
    "game_theory": "game_theory_claims",
}


# Per-analyst verification guidance: tells the verifier which facts can be
# cross-checked between sources and which must be re-computed from the
# re-fetched data (because the cross tools do NOT contain that fact).
_VERIFY_GUIDANCE: Dict[str, str] = {
    "fundamentals": (
        "交叉比对指引：\n"
        "- 业绩预告/快报/预测类数据：与【原始数据 [tool_news_sentiment]】的公告大全比对，"
        "确认对应公告真实存在且数值一致；\n"
        "- 报告中出现的最新价、市值换算：与【原始数据 [tool_technical]】日K最新收盘价比对；\n"
        "- 财务比率（ROIC/毛利率/净利率/ROE/EPS 等）：其他工具不含财务数据，"
        "必须使用【原始数据 [tool_fundamental]】的财务指标/三表重新计算核对（复算），禁止用价格数据比对财务比率。\n"
        "- 业绩预告/重大财务事件是否真实发布：用【原始数据 [web_search_tool]】的公开渠道检索结果验证；"
        "若 web_search 返回错误标记（未配置或检索失败），跳过基于它的校验。"
    ),
    "technical": (
        "交叉比对指引：\n"
        "- 资金流向类数据：与【原始数据 [tool_game_theory]】的个股资金流向比对，两源应一致；\n"
        "- 涨跌幅/涨跌停/高低点：与【原始数据 [tool_game_theory]】的涨跌停价格、"
        "【原始数据 [tool_technical]】日K 比对；\n"
        # "- 趋势/动量/市场环境结论：与【原始数据 [tool_special_data]】的趋势判断/市场环境评分做一致性校验；\n"  # 已停用
        "- 均线/指标数值（MA/MACD/RSI/BOLL 等）：用【原始数据 [tool_technical]】日K 重新计算核对。\n"
        "- 停牌/除权除息/重大新闻等影响价格的事件：用【原始数据 [web_search_tool]】的公开渠道检索结果与 K 线交叉验证；"
        "若 web_search 返回错误标记（未配置或检索失败），跳过基于它的校验。"
    ),
    "game_theory": (
        "交叉比对指引：\n"
        "- 内部人交易、高管增减持：与【原始数据 [tool_risk]】比对——两个独立端点返回同一事实，数值应一致；\n"
        "- 资金流向类数据：与【原始数据 [tool_technical]】的个股资金流向比对；\n"
        "- 龙虎榜/大宗交易/质押/两融明细：用【原始数据 [tool_game_theory]】复拉结果复算与核验。\n"
        "- 减持/质押/龙虎榜等公告是否真实发布：用【原始数据 [web_search_tool]】的公开渠道检索结果验证；"
        "若 web_search 返回错误标记（未配置或检索失败），跳过基于它的校验。"
    ),
}


def _build_verify_prompt(
    analyst_type: str,
    report: str,
    claims: list,
    raw_data: Dict[str, str],
) -> str:
    """Assemble the verifier prompt: report + claims + freshly fetched data."""
    parts = [
        "你是一名数据校验员。下面是 " + analyst_type + " 分析师的报告及其声明的数据来源。",
        "请对报告中每一个关键数据点执行校验：",
        "1. 事实类数据：将报告中的值与下方【原始数据】比对，判断是否一致（允许合理精度误差）。",
        "2. 计算类数据（比率、均值、涨跌幅、指标等）：使用下方【原始数据】重新计算核对。",
        "3. 只有当下方【原始数据】确实包含该事实时才可交叉比对；不包含该事实的数据源不得用于比对该项。",
        _VERIFY_GUIDANCE.get(analyst_type, ""),
        "逐项输出 FactVerificationReport：",
        "- passed=False 的项必须在 failure_reason 中说明具体差异，并在 feedback 中给出可操作的修正指引；",
        "- 只有全部通过时 overall_passed 才为 True；",
        "- 报告中的数据若在原始数据中找不到依据，必须标记为未通过（数据无来源）。",
        "",
        "### 报告全文",
        report,
        "",
        "### 结构化 claims（来源声明）",
        (
            json.dumps(claims, ensure_ascii=False, indent=2)
            if claims
            else "（无结构化 claims，请从报告文本中提取所有关键数据点进行校验）"
        ),
    ]
    for tool_name, data in raw_data.items():
        parts.append(f"\n### 原始数据 [{tool_name}]（本次重新获取）\n{data[:8000]}")
    return "\n".join(parts)


def _build_search_plan_prompt(
    analyst_type: str,
    report: str,
    claims: list,
    ts_code: str,
    previous_results: Dict[str, str] | None = None,
) -> str:
    """Prompt for the search-planning LLM (one planning round)."""
    parts = [
        "你负责为 " + analyst_type + " 分析师的报告规划联网搜索，用于事实校验。",
        "标的代码：" + ts_code,
        "任务：列出需要执行的搜索查询（0-N 条）。每一条应针对报告中一个需要在公开渠道验证的事实"
        "（如业绩预告/财报公告、减持/质押/龙虎榜、停牌/除权等事件、重大新闻）。",
        "要求：",
        "- query 建议格式：site:eastmoney.com <股票代码> <关键词>；",
        "- 已有内部数据源能验证的事实（如可复算的财务比率、双源资金流向比对）不需要搜索；",
        "- 不要搜索无法在公开渠道确认的推测性内容。",
    ]
    if previous_results:
        parts.append(
            "这是第 2 轮规划：基于已有搜索结果，列出仍需补充的搜索查询。"
            "若已有结果已足够，返回空列表。"
        )
        parts.append("### 已有搜索结果")
        for key, data in previous_results.items():
            parts.append(f"--- [{key}]\n{data[:1500]}")
    parts.extend([
        "",
        "### 报告全文",
        report,
        "",
        "### 结构化 claims（来源声明）",
        (
            json.dumps(claims, ensure_ascii=False, indent=2)
            if claims
            else "（无结构化 claims，请从报告文本中识别需要公开渠道验证的事实）"
        ),
    ])
    return "\n".join(parts)


def _run_web_search_rounds(
    plan_llm: Any,
    analyst_type: str,
    report: str,
    claims: list,
    ts_code: str,
    max_queries: int = 4,
    max_rounds: int = 2,
) -> Dict[str, str]:
    """Multi-round web search: plan -> search -> (re-plan) -> done.

    Each planning round asks the LLM for the queries to run; every query is
    executed in code via web_search_tool (one retry when the provider returns
    an error/empty marker). Returns a dict keyed by web_search:<query>.

    When the planner is unavailable or fails, a single template query for the
    analyst's theme is used so the cross-source check still happens.
    """
    from tradingagents.agents.schemas import VerificationSearchPlan

    results: Dict[str, str] = {}
    if plan_llm is None:
        template = _WEB_SEARCH_QUERIES.get(analyst_type, "").format(ts_code=ts_code)
        if template:
            results[f"web_search:{template}"] = _search_with_retry(template)
        return results

    base_prompt = _build_search_plan_prompt(analyst_type, report, claims, ts_code)
    for round_index in range(max_rounds):
        try:
            if results:
                prompt = _build_search_plan_prompt(
                    analyst_type, report, claims, ts_code, previous_results=results
                )
            else:
                prompt = base_prompt
            plan = plan_llm.invoke(prompt)
            queries = [q for q in (plan.queries or []) if q and q.strip()]
        except Exception as exc:
            logger.warning(
                "FactChecker-%s: search planning failed (%s); using template query",
                analyst_type, exc,
            )
            template = _WEB_SEARCH_QUERIES.get(analyst_type, "").format(ts_code=ts_code)
            queries = [template] if template else []

        queries = queries[:max_queries]
        added = False
        for query in queries:
            key = f"web_search:{query[:60]}"
            if key in results:
                continue
            results[key] = _search_with_retry(query)
            added = True
        if not added or not queries:
            break
    return results


def _search_with_retry(query: str) -> str:
    """Run one web search; retry once when the provider returns a marker."""
    result = web_search_tool.invoke({"query": query, "max_output": 2000})
    if result.startswith("[web_search]"):
        retry = web_search_tool.invoke({"query": query, "max_output": 2000})
        if not retry.startswith("[web_search]"):
            return retry
    return result


def create_fact_checker(
    analyst_type: str,
    verify_llm: Any,
    max_rounds: int = 2,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create the fact-checker node for one analyst.

    Args:
        analyst_type: fundamentals / technical / game_theory.
        verify_llm: LLM used for the structured verification pass.
        max_rounds: Retry budget before the report is marked unverified.
    """
    structured_llm = bind_structured(
        verify_llm, FactVerificationReport, f"FactChecker-{analyst_type}"
    )
    from tradingagents.agents.schemas import VerificationSearchPlan

    plan_llm = bind_structured(
        verify_llm, VerificationSearchPlan, f"SearchPlanner-{analyst_type}"
    )

    def fact_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
        report_key = _REPORT_KEYS[analyst_type]
        claim_key = _CLAIM_KEYS[analyst_type]
        report = state.get(report_key, "") or ""
        claims = state.get(claim_key, []) or []

        vs = (state.get("verification_state") or {}).get(analyst_type) or {}
        attempts = int(vs.get("attempts", 0)) + 1

        # 1) Code-level re-fetch: own tool + cross-source tools.
        #    web_search_tool is declared in CROSS_VERIFY_TOOLS as a cross
        #    source, but it is executed ONLY by the multi-round planner below
        #    (LLM-planned queries, budgeted rounds, retry) — never here — so a
        #    template search and a planned search never both fire.
        web_search_on = bool(get_config().get("web_search_enabled", False))
        raw_data: Dict[str, str] = {}
        for tool in CROSS_VERIFY_TOOLS[analyst_type]:
            if tool.name == "web_search_tool":
                continue
            try:
                raw_data[tool.name] = tool.invoke(
                    {
                        "ts_code": state["company_of_interest"],
                        "end_date": state.get("trade_date", ""),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - a tool failure must not kill the stage
                raw_data[tool.name] = f"[取数失败] {exc}"

        # 2) Multi-round web search (LLM-planned, budgeted): verify facts that
        #    need public-channel confirmation (announcements, events, news).
        if web_search_on:
            cfg = get_config()
            raw_data.update(
                _run_web_search_rounds(
                    plan_llm,
                    analyst_type,
                    report,
                    claims,
                    state["company_of_interest"],
                    max_queries=int(cfg.get("verify_search_max_queries", 4)),
                    max_rounds=int(cfg.get("verify_search_max_rounds", 2)),
                )
            )

        # 3) Structured verification pass (single LLM call when structured
        #    output is available; free-text fallback otherwise)
        prompt = _build_verify_prompt(analyst_type, report, claims, raw_data)
        passed = False
        feedback = ""
        items: list = []
        verification_md = ""
        if structured_llm is not None:
            try:
                result = structured_llm.invoke(prompt)
                passed = bool(result.overall_passed)
                feedback = result.feedback or ""
                items = [i.model_dump() for i in result.items]
                verification_md = render_verification_report(result)
            except Exception as exc:
                logger.warning(
                    "FactChecker-%s: structured verification failed (%s); "
                    "falling back to free-text verdict",
                    analyst_type, exc,
                )
                verification_md = invoke_structured_or_freetext(
                    None, verify_llm, prompt, render_verification_report,
                    f"FactChecker-{analyst_type}",
                )
                passed = "PASSED" in verification_md.upper() and "FAILED" not in verification_md.upper()
                feedback = verification_md
        else:
            verification_md = invoke_structured_or_freetext(
                None, verify_llm, prompt, render_verification_report,
                f"FactChecker-{analyst_type}",
            )
            passed = "PASSED" in verification_md.upper() and "FAILED" not in verification_md.upper()
            feedback = verification_md

        new_vs = {
            "attempts": attempts,
            "passed": bool(passed),
            "feedback": feedback,
            "items": items,
            "report_md": verification_md,
        }
        # Return ONLY this analyst's key: the channel has a merge reducer so
        # the parallel fact-checkers update it concurrently without clobbering
        # each other (last-write-wins per analyst).
        logger.info(
            "FactChecker-%s: attempt %d/%d %s (%d claims, %d items)",
            analyst_type, attempts, max_rounds,
            "PASSED" if passed else "FAILED", len(claims), len(items),
        )
        return {"verification_state": {analyst_type: new_vs}}

    return fact_checker_node


def make_verify_router(
    analyst_type: str,
    max_rounds: int = 2,
) -> Callable[[Dict[str, Any]], str]:
    """Create the conditional router placed after a fact-checker node.

    Returns "retry" when the verification failed and the retry budget is not
    exhausted, "next" otherwise (passed, or budget exhausted -> continue with
    the report marked unverified).
    """

    def router(state: Dict[str, Any]) -> str:
        vs = (state.get("verification_state") or {}).get(analyst_type) or {}
        if vs.get("passed"):
            return "next"
        if int(vs.get("attempts", 0)) >= max_rounds:
            return "next"
        return "retry"

    return router
