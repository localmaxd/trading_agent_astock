from tradingagents.llm_clients.openai_client import OpenAIClient
from dotenv import load_dotenv

load_dotenv()

client = OpenAIClient("deepseek-chat", provider="deepseek")
llm = client.get_llm()
result = llm.invoke([("human", "Say hello")])
print(result)
