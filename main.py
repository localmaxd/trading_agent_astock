from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "local"
config["deep_think_llm"] = "Qwen3.6-35B-A3B-8bit"
config["quick_think_llm"] = "Qwen3.6-35B-A3B-8bit"
config["backend_url"] = "http://127.0.0.1:8091/v1"
config["max_debate_rounds"] = 1
config["output_language"] = "Chinese"

# 5 analysts for A-share analysis: 基本面, 技术面, 博弈面, 风险面, 新闻舆情
config["selected_analysts"] = [
    "fundamentals",
    "technical",
    "game_theory",
    "risk",
    "news_sentiment",
]

# Initialize with custom config
ta = TradingAgentsGraph(
    selected_analysts=config["selected_analysts"],
    debug=True,
    config=config,
)

# forward propagate — use A-share stock code
# Example: 600519 = 贵州茅台 (Kweichow Moutai)
_, decision = ta.propagate("600519", "2024-05-10")
print(decision)
