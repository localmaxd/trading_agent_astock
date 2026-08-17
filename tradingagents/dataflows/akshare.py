"""akshare data vendor for A-share stock analysis.

Provides wrapper functions organized into 5 categories matching the
5 analyst agents. Each function accepts stock_code and applies
per-interface filtering logic.
"""

import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import akshare as ak

MAX_ROWS = 40
MAX_CHARS = 12000


def _retry(fn, max_tries=3, delay=2):
    """Retry fn on ConnectionError up to max_tries times."""
    last_err = None
    for attempt in range(max_tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if "Connection" in type(e).__name__ or "RemoteDisconnected" in str(e):
                if attempt < max_tries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise
    raise last_err


def _fmt(df: pd.DataFrame, name: str, total_rows: int = None) -> str:
    """Convert DataFrame to truncated CSV string with header."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return f"[{name}] 查询成功，但返回数据为空"
    if total_rows is None:
        total_rows = len(df)
    n = len(df)
    if n > MAX_ROWS:
        df = df.head(MAX_ROWS)
    result = (
        f"# {name}\n"
        f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Rows: {total_rows}\n\n"
        f"{df.to_csv(index=False)}"
    )
    if len(result) > MAX_CHARS:
        result = result[:MAX_CHARS] + (
            f"\n\n[... 已截断, 原始{len(result)}字符, {total_rows}行]"
        )
    elif total_rows > MAX_ROWS:
        result += f"\n\n[... 仅展示前{MAX_ROWS}行, 共{total_rows}行]"
    return result


def _to_shsz(code: str) -> str:
    """'000001' -> 'SZ000001', '600519' -> 'SH600519'"""
    code = str(code).strip().upper()
    digits = _extract_digits(code)
    if digits.startswith("6") or digits.startswith("5"):
        return f"SH{digits}"
    return f"SZ{digits}"


def _to_dot_fmt(code: str) -> str:
    """'000001' -> '000001.SZ', '600519' -> '600519.SH'"""
    code = str(code).strip().upper()
    digits = _extract_digits(code)
    if digits.startswith("6") or digits.startswith("5"):
        return f"{digits}.SH"
    return f"{digits}.SZ"


def _to_shsz_lower(code: str, market: str = "") -> str:
    """'000001' -> 'sz000001', '600519' -> 'sh600519'"""
    code = str(code).strip().upper()
    digits = _extract_digits(code)
    if market:
        return f"{market}{digits}"
    if digits.startswith("6") or digits.startswith("5"):
        return f"sh{digits}"
    return f"sz{digits}"


def _extract_digits(stock_code: str) -> str:
    """'SH600519' -> '600519', '600519.SH' -> '600519'"""
    import re
    m = re.search(r'(\d{6})', str(stock_code))
    return m.group(1) if m else str(stock_code)


def _recent_quarters(n: int = 8) -> list:
    """Return last n quarter-end dates as YYYYMMDD strings."""
    now = datetime.now()
    quarters = []
    for i in range(n):
        y = now.year
        m = ((now.month - 1) // 3) * 3 + 1  # 1,4,7,10
        d = datetime(y, m, 1) - timedelta(days=1)
        for _ in range(i):
            d = datetime(d.year, ((d.month - 1) // 3) * 3 + 1, 1) - timedelta(days=1)
        quarters.append(d.strftime("%Y%m%d"))
    return quarters


# ═══════════════════════════════════════════════════════════════════════════════
# 一、基本面 (Fundamentals) — 9 interfaces
# ═══════════════════════════════════════════════════════════════════════════════

# 财务核心指标精选列
_FIN_INDICATOR_COLS = {
    "SECURITY_CODE": "代码", "SECURITY_NAME_ABBR": "简称",
    "REPORT_DATE": "报告期", "REPORT_TYPE": "类型",
    "EPSJB": "每股收益", "EPSKCJB": "扣非每股收益",
    "BPS": "每股净资产", "MGZBGJ": "每股资本公积", "MGWFPLR": "每股未分利润",
    "TOTALOPERATEREVE": "营业总收入", "PARENTNETPROFIT": "归母净利润",
    "KCFJCXSYJLR": "扣非净利润", "MLR": "毛利",
    "ROEJQ": "ROE(%)", "ROEKCJQ": "扣非ROE(%)",
    "XSMLL": "销售毛利率(%)", "XSJLL": "销售净利率(%)",
    "YYZSRGDHBZC": "营收同比(%)", "NETPROFITRPHBZC": "净利润同比(%)",
    "KFJLRGDHBZC": "扣非净利同比(%)",
    "ZCFZL": "资产负债率(%)", "LD": "流动比率", "SD": "速动比率",
    "JYXJLYYSR": "经营现金流/营收", "XJLLB": "现金流净额",
    "ROIC": "ROIC(%)",
}

# 资产负债表精选列
_BS_COLS = {
    "SECURITY_CODE": "代码", "SECURITY_NAME_ABBR": "简称",
    "REPORT_DATE": "报告期", "REPORT_TYPE": "类型",
    "TOTAL_ASSETS": "总资产", "TOTAL_LIABILITIES": "总负债",
    "TOTAL_EQUITY": "股东权益", "CURRENT_ASSETS": "流动资产",
    "CURRENT_LIABILITIES": "流动负债", "FIXED_ASSETS": "固定资产",
    "MONEY_CAPITAL": "货币资金", "ACCOUNTS_RECEIVABLE": "应收账款",
    "INVENTORY": "存货", "GOODWILL": "商誉",
    "SHORT_LOANS": "短期借款", "LONG_LOANS": "长期借款",
    "BOND_PAYABLE": "应付债券", "MINORITY_INTERESTS": "少数股东权益",
    "TOTAL_CAPITAL_STOCK": "总股本", "UNDISTRIBUTED_PROFIT": "未分利润",
}

# 利润表精选列
_IS_COLS = {
    "SECURITY_CODE": "代码", "SECURITY_NAME_ABBR": "简称",
    "REPORT_DATE": "报告期", "REPORT_TYPE": "类型",
    "TOTAL_OPERATING_REVENUE": "营业收入", "OPERATING_COST": "营业成本",
    "SALES_EXPENSE": "销售费用", "ADMIN_EXPENSE": "管理费用",
    "RESEARCH_EXPENSE": "研发费用", "FINANCIAL_EXPENSE": "财务费用",
    "OPERATING_PROFIT": "营业利润", "TOTAL_PROFIT": "利润总额",
    "NET_PROFIT": "净利润", "PARENT_NET_PROFIT": "归母净利润",
    "MINORITY_INTEREST": "少数股东损益",
    "BASIC_EPS": "基本每股收益", "DILUTED_EPS": "稀释每股收益",
    "OTHER_COMPRE_INCOME": "其他综合收益", "TOTAL_COMPRE_INCOME": "综合收益总额",
}

# 现金流量表精选列
_CF_COLS = {
    "SECURITY_CODE": "代码", "SECURITY_NAME_ABBR": "简称",
    "REPORT_DATE": "报告期", "REPORT_TYPE": "类型",
    "NET_OPERATING_CASH_FLOW": "经营活动净现金流",
    "NET_INVESTING_CASH_FLOW": "投资活动净现金流",
    "NET_FINANCING_CASH_FLOW": "筹资活动净现金流",
    "FREE_CASH_FLOW": "自由现金流",
    "SALES_GOODS_CASH_FLOW": "销售商品收到现金",
    "PURCHASE_GOODS_CASH_FLOW": "购买商品支付现金",
    "NET_CASH_INCREASE": "现金净增加额",
    "CCE_AT_BEGINNING": "期初现金等价物", "CCE_AT_END": "期末现金等价物",
}

# 实时行情精选列
_SPOT_COLS = {
    "代码": "代码", "名称": "简称", "最新价": "最新价", "涨跌幅": "涨跌幅(%)",
    "涨跌额": "涨跌额", "成交量": "成交量", "成交额": "成交额",
    "振幅": "振幅(%)", "换手率": "换手率(%)", "量比": "量比",
    "市盈率-动态": "动态PE", "市净率": "PB", "60日涨跌幅": "60日涨跌(%)",
    "年初至今涨跌幅": "年初至今涨跌(%)",
}


def _select_cols(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Select and rename columns; fall back to all columns if <4 match."""
    available = [c for c in col_map if c in df.columns]
    if len(available) >= 4:
        return df[available].rename(columns={c: col_map[c] for c in available})
    # Column names don't match (e.g. bank vs general industry) — return raw
    return df


def get_financial_analysis_indicator(stock_code: str) -> str:
    """财务核心指标速查 — 近两年按报告期，仅返回精选核心列"""
    code = _to_dot_fmt(stock_code)
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=code, indicator="按报告期")
        if df is None or df.empty:
            return f"[财务指标] 数据为空"
        df = df.head(8) if len(df) > 8 else df
        df = _select_cols(df, _FIN_INDICATOR_COLS)
        return _fmt(df, "财务核心指标(近两年)", len(df))
    except Exception as e:
        return f"[财务指标] 失败: {e}"


def get_balance_sheet_report(stock_code: str) -> str:
    """资产负债表 — 近两年，精选列"""
    code = _to_shsz(stock_code)
    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=code)
        if df is None or df.empty:
            return f"[资产负债表] 数据为空"
        df = df.head(8) if len(df) > 8 else df
        df = _select_cols(df, _BS_COLS)
        return _fmt(df, "资产负债表(近两年)", len(df))
    except Exception as e:
        return f"[资产负债表] 失败: {e}"


def get_profit_sheet_report(stock_code: str) -> str:
    """利润表 — 近两年，精选列"""
    code = _to_shsz(stock_code)
    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=code)
        if df is None or df.empty:
            return f"[利润表] 数据为空"
        df = df.head(8) if len(df) > 8 else df
        df = _select_cols(df, _IS_COLS)
        return _fmt(df, "利润表(近两年)", len(df))
    except Exception as e:
        return f"[利润表] 失败: {e}"


def get_cash_flow_sheet_report(stock_code: str) -> str:
    """现金流量表 — 近两年，精选列"""
    code = _to_shsz(stock_code)
    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
        if df is None or df.empty:
            return f"[现金流量表] 数据为空"
        df = df.head(8) if len(df) > 8 else df
        df = _select_cols(df, _CF_COLS)
        return _fmt(df, "现金流量表(近两年)", len(df))
    except Exception as e:
        return f"[现金流量表] 失败: {e}"


def get_performance_preview(stock_code: str, date: str = "20260331") -> str:
    """业绩预告 — 筛选该股票在指定报告期的预告"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_yjyg_em(date=date)
        if df is None or df.empty:
            return f"[业绩预告] 数据为空"
        # 筛选该股票
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[业绩预告] 未找到股票代码 {digits} 的预告数据(date={date})"
        return _fmt(filtered, f"业绩预告-{digits}", len(filtered))
    except Exception as e:
        return f"[业绩预告] 失败: {e}"


def get_performance_express(stock_code: str, date: str = "20260331") -> str:
    """业绩快报 — 筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_yjkb_em(date=date)
        if df is None or df.empty:
            return f"[业绩快报] 数据为空"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[业绩快报] 未找到 {digits} 的快报数据(date={date})"
        return _fmt(filtered, f"业绩快报-{digits}", len(filtered))
    except Exception as e:
        return f"[业绩快报] 失败: {e}"


def get_profit_forecast(stock_code: str) -> str:
    """机构一致预期 — 全量拉取后筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_profit_forecast_em(symbol="")
        if df is None or df.empty:
            return f"[机构一致预期] 数据为空"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[机构一致预期] 未找到 {digits} 的预测数据"
        return _fmt(filtered, f"机构一致预期-{digits}", len(filtered))
    except Exception as e:
        return f"[机构一致预期] 失败: {e}"


def get_main_business_composition(stock_code: str) -> str:
    """主营构成 — 近两年"""
    code = _to_shsz(stock_code)
    try:
        df = ak.stock_zygc_em(symbol=code)
        if df is None or df.empty:
            return f"[主营构成] 数据为空"
        df = df.head(8) if len(df) > 8 else df
        return _fmt(df, "主营构成(近两年)", len(df))
    except Exception as e:
        return f"[主营构成] 失败: {e}"


def get_industry_pe_ratio(date: str = "20260331") -> str:
    """行业估值锚 — 全量返回"""
    try:
        df = ak.stock_industry_pe_ratio_cninfo(symbol="证监会行业分类", date=date)
        if df is None or df.empty:
            return f"[行业PE] 数据为空"
        return _fmt(df, f"行业PE(date={date})", len(df))
    except Exception as e:
        return f"[行业PE] 失败: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 二、技术面 (Technical) — 5 interfaces (删除了4个)
# ═══════════════════════════════════════════════════════════════════════════════

def get_a_spot() -> str:
    """实时行情快照 — 精选列"""
    def _call():
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return f"[实时行情] 数据为空"
        total = len(df)
        df = _select_cols(df, _SPOT_COLS)
        return _fmt(df, f"实时行情(共{total}只)", len(df))
    try:
        return _retry(_call)
    except Exception as e:
        return f"[实时行情] 失败: {e}"


def _baostock_query(code: str, fields: str, start_date: str, end_date: str,
                    frequency: str, adjustflag: str, max_tries: int = 3):
    """baostock query with retry on decompression errors."""
    import baostock as bs
    for attempt in range(max_tries):
        lg = bs.login()
        try:
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag=adjustflag)
            if rs.error_code != '0':
                return None, f"查询失败: {rs.error_msg}"
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            return rows, ""
        except Exception as e:
            if attempt < max_tries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return None, str(e)
        finally:
            try:
                bs.logout()
            except Exception:
                pass


def _baostock_daily(stock_code: str, start_date: str, end_date: str, adjust: str = "3") -> str:
    """baostock 日K线 fallback。adjust: 1=后复权 2=前复权 3=不复权"""
    digits = _extract_digits(stock_code)
    code = f"sh.{digits}" if digits.startswith("6") else f"sz.{digits}"
    if len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    rows, err = _baostock_query(code, fields, start_date, end_date, "d", adjust)
    if err:
        return f"[日K线-baostock] {err}"
    if not rows:
        return f"[日K线-baostock] {stock_code} {start_date}~{end_date} 无数据"
    df = pd.DataFrame(rows, columns=fields.split(","))
    return _fmt(df, f"日K线-baostock-{digits}", len(df))


def get_a_hist(stock_code: str, start_date: str, end_date: str, adjust: str = "qfq") -> str:
    """历史日频K线 — 优先akshare，失败回退baostock"""
    # map adjust to baostock format
    baostock_adj = {"qfq": "2", "hfq": "1", "": "3"}.get(adjust, "2")
    def _call_ak():
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                start_date=start_date, end_date=end_date, adjust=adjust)
        if df is None or df.empty:
            return None
        return _fmt(df, f"日K线-{stock_code}", len(df))
    try:
        r = _retry(_call_ak)
        if r is not None:
            return r
    except Exception:
        pass
    # Fallback to baostock
    try:
        return _baostock_daily(stock_code, start_date, end_date, baostock_adj)
    except Exception as e:
        return f"[日K线] akshare和baostock均失败: {e}"


def _baostock_min(stock_code: str, start_date: str, end_date: str, period: str = "5") -> str:
    """baostock 分钟K线 fallback（带重试）"""
    digits = _extract_digits(stock_code)
    code = f"sh.{digits}" if digits.startswith("6") else f"sz.{digits}"
    if len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    rows, err = _baostock_query(code, fields, start_date, end_date, period, "3")
    if err:
        return f"[分钟K线-baostock] {err}"
    if not rows:
        return f"[分钟K线-baostock] {stock_code} 无数据"
    df = pd.DataFrame(rows, columns=fields.split(","))
    return _fmt(df, f"分钟K线-baostock-{digits}-{period}min", len(df))


def get_a_hist_min(stock_code: str, start_date: str, end_date: str,
                   period: str = "5", adjust: str = "") -> str:
    """分钟级分时 — 优先akshare，失败回退baostock"""
    if len(start_date) == 8:
        start_date_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 09:30:00"
    else:
        start_date_fmt = start_date
    if len(end_date) == 8:
        end_date_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]} 15:00:00"
    else:
        end_date_fmt = end_date
    def _call_ak():
        df = ak.stock_zh_a_hist_min_em(symbol=stock_code, start_date=start_date_fmt,
                                       end_date=end_date_fmt, period=period, adjust=adjust)
        if df is None or df.empty:
            return None
        return _fmt(df, f"分钟K线-{stock_code}-{period}min", len(df))
    try:
        r = _retry(_call_ak)
        if r is not None:
            return r
    except Exception:
        pass
    try:
        return _baostock_min(stock_code, start_date, end_date, period)
    except Exception as e:
        return f"[分钟K线] akshare和baostock均失败: {e}"


def get_individual_fund_flow(stock_code: str) -> str:
    """个股资金流向"""
    digits = _extract_digits(stock_code)
    market = "sh" if digits.startswith("6") or digits.startswith("5") else "sz"
    try:
        df = ak.stock_individual_fund_flow(stock=digits, market=market)
        if df is None or df.empty:
            return f"[资金流向] 无数据"
        return _fmt(df, f"资金流向-{digits}", len(df))
    except Exception as e:
        return f"[资金流向] 失败: {e}"


def get_market_congestion() -> str:
    """大盘拥挤度 — 取最近8条"""
    try:
        df = ak.stock_a_congestion_lg()
        if df is None or df.empty:
            return f"[拥挤度] 数据为空"
        df = df.head(8) if len(df) > 8 else df
        return _fmt(df, "大盘拥挤度(最近8条)", len(df))
    except Exception as e:
        return f"[拥挤度] 失败: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 三、博弈面 (Game Theory) — 9 interfaces (删除了2个, 新增过滤逻辑)
# ═══════════════════════════════════════════════════════════════════════════════

def get_lhb_detail(stock_code: str, start_date: str, end_date: str) -> str:
    """龙虎榜详情 — 最近7天，筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return f"[龙虎榜] {start_date}~{end_date} 无数据"
        # 筛选该股票
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[龙虎榜] {digits} 在 {start_date}~{end_date} 无上榜记录"
        return _fmt(filtered, f"龙虎榜-{digits}", len(filtered))
    except Exception as e:
        return f"[龙虎榜] 失败: {e}"


def get_lhb_institution_stat(stock_code: str, period: str = "近一月") -> str:
    """机构席位追踪 — 筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_lhb_jgstatistic_em(symbol=period)
        if df is None or df.empty:
            return f"[机构席位] 无数据"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[机构席位] {digits} 在{period}内无记录"
        return _fmt(filtered, f"机构席位-{digits}({period})", len(filtered))
    except Exception as e:
        return f"[机构席位] 失败: {e}"


def get_management_change(stock_code: str) -> str:
    """高管增减持"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_management_change_ths(symbol=digits)
        if df is None or df.empty:
            return f"[高管增减持] {digits} 无数据或近期无变动"
        return _fmt(df, f"高管增减持-{digits}", len(df))
    except Exception as e:
        return f"[高管增减持] 失败: {e}"


def get_institute_hold_detail(stock_code: str) -> str:
    """机构持股详情 — 过去4个季度"""
    digits = _extract_digits(stock_code)
    now = datetime.now()
    quarters = []
    for i in range(4):
        q_month = ((now.month - 1) // 3) * 3 + 1
        y = now.year
        for _ in range(i):
            q_month -= 3
            if q_month < 1:
                q_month = 10
                y -= 1
        quarters.append(f"{y}{q_month}1")  # e.g. 20261
    results = []
    for q in quarters:
        try:
            df = ak.stock_institute_hold_detail(stock=digits, quarter=q)
            if df is not None and not df.empty:
                results.append(f"--- {q}季度 ---\n{df.head(15).to_csv(index=False)}")
        except Exception:
            pass
    if not results:
        return f"[机构持股] {digits} 过去4季度无数据"
    return "\n\n".join(results)[:MAX_CHARS]


def get_fund_report_stock(stock_code: str) -> str:
    """基金重仓股 — 该股票的基金覆盖排名及近半年变化"""
    digits = _extract_digits(stock_code)
    # 获取最近两个报告期
    now = datetime.now()
    dates = []
    for offset_months in [0, 6]:
        d = now - timedelta(days=offset_months * 30)
        q_month = ((d.month - 1) // 3) * 3 + 1
        dates.append(f"{d.year}{q_month:02d}30")
    results = []
    for dt in dates:
        try:
            df = ak.fund_report_stock_cninfo(date=dt)
            if df is not None and not df.empty:
                mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
                fdf = df[mask]
                if not fdf.empty:
                    results.append(f"--- 报告期 {dt} ---\n{fdf.head(10).to_csv(index=False)}")
        except Exception:
            pass
    if not results:
        return f"[基金重仓] {digits} 无基金覆盖数据"
    out = "\n\n".join(results)
    return out[:MAX_CHARS]


def get_institute_research_stat(stock_code: str, date: str = "20260517") -> str:
    """机构调研统计 — 筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_jgdy_tj_em(date=date)
        if df is None or df.empty:
            return f"[机构调研] 无数据"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[机构调研] {digits} 近期无调研记录"
        return _fmt(filtered, f"机构调研-{digits}", len(filtered))
    except Exception as e:
        return f"[机构调研] 失败: {e}"


def get_block_trade_detail(stock_code: str, start_date: str, end_date: str) -> str:
    """大宗交易明细 — 前一天到今天 + 筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_dzjy_mrmx(symbol="基金", start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return f"[大宗交易] 无数据"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[大宗交易] {digits} 近期无大宗交易"
        return _fmt(filtered, f"大宗交易-{digits}", len(filtered))
    except Exception as e:
        return f"[大宗交易] 失败: {e}"


def get_inner_trade(stock_code: str) -> str:
    """内部人交易 — 筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_inner_trade_xq()
        if df is None or df.empty:
            return f"[内部人交易] 无数据"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[内部人交易] {digits} 无近期交易记录"
        return _fmt(filtered, f"内部人交易-{digits}", len(filtered))
    except Exception as e:
        return f"[内部人交易] 失败: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 四、风险面 (Risk) — 3 interfaces (删除了3个)
# ═══════════════════════════════════════════════════════════════════════════════

def get_goodwill_balance(stock_code: str, date: str = "20260331") -> str:
    """个股商誉余额 — 筛选该股票"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_sy_em(date=date)
        if df is None or df.empty:
            return f"[商誉余额] 数据为空"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[商誉余额] 未找到 {digits} 的商誉数据"
        return _fmt(filtered, f"商誉余额-{digits}", len(filtered))
    except Exception as e:
        return f"[商誉余额] 失败: {e}"


def get_restricted_release_detail(stock_code: str, start_date: str = None,
                                  end_date: str = None) -> str:
    """限售解禁详情 — 近三个月，筛选该股票"""
    digits = _extract_digits(stock_code)
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
    try:
        df = ak.stock_restricted_release_detail_em(start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return f"[限售解禁] 近三个月无数据"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[限售解禁] {digits} 近三个月无解禁计划"
        return _fmt(filtered, f"限售解禁-{digits}(近三月)", len(filtered))
    except Exception as e:
        return f"[限售解禁] 失败: {e}"


def get_st_risk(stock_code: str) -> str:
    """ST风险警示 — 检查该股票是否在ST名单中"""
    digits = _extract_digits(stock_code)
    def _call():
        df = ak.stock_zh_a_st_em()
        if df is None or df.empty:
            return f"[ST风险] 数据为空"
        mask = df.astype(str).apply(lambda row: row.str.contains(digits).any(), axis=1)
        filtered = df[mask]
        if filtered.empty:
            return f"[ST风险] ✓ {digits} 不在当前ST/*ST名单中"
        return _fmt(filtered, f"ST风险-{digits} ⚠️ 在ST名单中!", len(filtered))
    try:
        return _retry(_call)
    except Exception as e:
        return f"[ST风险] 失败: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 五、新闻舆情 (News & Sentiment) — 5 interfaces (删除了1个)
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_news(stock_code: str) -> str:
    """个股新闻"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_news_em(symbol=digits)
        if df is None or df.empty:
            return f"[个股新闻] {digits} 无新闻"
        return _fmt(df, f"个股新闻-{digits}", len(df))
    except Exception as e:
        return f"[个股新闻] 失败: {e}"


def get_research_report(stock_code: str) -> str:
    """个股研报 — 取前10条"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_research_report_em(symbol=digits)
        if df is None or df.empty:
            return f"[研报] {digits} 无研报"
        df = df.head(10) if len(df) > 10 else df
        return _fmt(df, f"研报-{digits}(前10)", len(df))
    except Exception as e:
        return f"[研报] 失败: {e}"


def get_notice_report(date: str = "20240510") -> str:
    """公告大全"""
    try:
        df = ak.stock_notice_report(symbol="全部", date=date)
        if df is None or df.empty:
            return f"[公告] 无数据"
        return _fmt(df, f"公告(date={date})", len(df))
    except Exception as e:
        return f"[公告] 失败: {e}"


def get_irm_answers(stock_code: str) -> str:
    """互动易问答 — 直接按股票代码查询"""
    digits = _extract_digits(stock_code)
    try:
        df = ak.stock_irm_cninfo(symbol=digits)
        if df is None or df.empty:
            return f"[互动易] {digits} 无互动问答记录"
        df = df.head(10) if len(df) > 10 else df
        return _fmt(df, f"互动易-{digits}(前10)", len(df))
    except Exception as e:
        return f"[互动易] {digits} 无互动问答数据({type(e).__name__})"


def get_hot_rank() -> str:
    """个股人气榜"""
    try:
        df = ak.stock_hot_rank_em()
        if df is None or df.empty:
            return f"[人气榜] 数据为空"
        return _fmt(df, "人气榜", len(df))
    except Exception as e:
        return f"[人气榜] 失败: {e}"
