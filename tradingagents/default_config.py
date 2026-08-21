import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    # ========== Path configuration ==========
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    "memory_log_max_entries": None,

    # ========== LLM core configuration ==========
    "llm_provider": "deepseek",
    "deep_think_llm": "deepseek-v4-flash",
    "quick_think_llm": "deepseek-v4-flash",
    "backend_url": None,

    # ========== Provider-specific reasoning parameters ==========
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,

    # ========== Runtime behavior ==========
    "checkpoint_enabled": False,
    "output_language": "Chinese",

    # ========== Debate depth ==========
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # ========== Web search (DeepSeek Responses API built-in search) ==========
    # Optional capability: analysts listed in web_search_analysts may call
    # web_search_tool (eastmoney.com etc.) to supplement their analysis.
    "web_search_enabled": False,
    "web_search_api_key": None,          # None -> DEEPSEEK_API_KEY env var
    "web_search_base_url": "https://api.deepseek.com",
    "web_search_model": "deepseek-chat",
    "web_search_analysts": ["fundamentals", "technical", "game_theory"],

    # ========== Fact verification (per-analyst fact-check guard) ==========
    # After fundamentals / technical / game_theory output, a fact-checker
    # node re-fetches cross-source data, verifies claims (facts + re-computed
    # calculations), and feeds failures back to the analyst to redo its work.
    "verify_enabled": True,
    "max_verify_rounds": 2,              # retries before marking as unverified
    # Per-verification web-search budget (only used when web_search_enabled):
    # how many queries per planning round, and how many planning rounds.
    "verify_search_max_queries": 4,
    "verify_search_max_rounds": 2,

    # ========== Data vendor routing (akshare = primary for A-shares) ==========
    "data_vendors": {
        "core_stock_apis": "akshare",
        "technical_indicators": "akshare",
        "fundamental_data": "akshare",
        "news_data": "akshare",
    },

    # Tool-level vendor overrides (higher priority than category-level)
    "tool_vendors": {},
}
