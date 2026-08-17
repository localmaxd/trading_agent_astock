# tests/check_env.py
import os

from dotenv import load_dotenv
load_dotenv()
print("=== 环境变量诊断 ===")
print(f"KIMI_API_KEY: {os.environ.get('KIMI_API_KEY', '【未设置】')}")
print(f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY', '【未设置】')}")
print(f"当前工作目录: {os.getcwd()}")