import time
from tradingagents.dataflows.akshare import get_restricted_release_detail
from tradingagents.dataflows.akshare import get_cash_flow_sheet_report,get_lhb_detail,get_research_report,get_performance_express,get_performance_preview
from tradingagents.dataflows.akshare import get_financial_analysis_indicator,get_balance_sheet_report,get_profit_sheet_report,get_cash_flow_sheet_report
from tradingagents.dataflows.akshare import get_profit_forecast,get_main_business_composition,get_industry_pe_ratio

from tradingagents.dataflows.akshare import get_market_congestion

from tradingagents.dataflows.akshare import get_lhb_detail,get_lhb_institution_stat,get_institute_hold_detail

from tradingagents.dataflows.akshare import get_fund_report_stock,get_institute_research_stat,get_block_trade_detail,get_inner_trade

from tradingagents.dataflows.akshare import get_goodwill_balance,get_st_risk,get_stock_news,get_research_report,get_notice_report

from tradingagents.dataflows.akshare import get_irm_answers,get_hot_rank,get_a_spot,get_a_hist
start_time = time.time()
# result = get_free_shareholder_change(date="20250429")
# result = get_performance_preview('20250331')
# result = get_institute_hold_detail("300394","20261")


result = get_hot_rank()
print(result)
end_time = time.time()


