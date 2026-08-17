"""测试 external API 5个接口"""
import requests
import sys

TS_CODE = sys.argv[1] if len(sys.argv) > 1 else "300438.SZ"
END_DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-05-21"
BASE = "http://localhost:8000/api/external"

endpoints = [
    ("基本面 /fundamental", f"{BASE}/fundamental/{TS_CODE}", {}),
    ("技术面 /technical", f"{BASE}/technical/{TS_CODE}", {"end_date": END_DATE}),
    ("博弈面 /game", f"{BASE}/game/{TS_CODE}", {}),
    ("风险面 /risk", f"{BASE}/risk/{TS_CODE}", {"end_date": END_DATE}),
    ("新闻舆情 /risk_sentiment", f"{BASE}/risk_sentiment/{TS_CODE}", {"end_date": END_DATE}),
    ("特殊数据 /special_data", f"{BASE}/special_data/{TS_CODE}", {"end_date": END_DATE}),
]

print(f"External API 测试 | ts_code={TS_CODE} end_date={END_DATE}")
print("=" * 60)

for name, url, params in endpoints:
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            content = r.json().get("content", "")
            print(f"\n--- {name} ---")
            print(f"状态: OK, 长度: {len(content)}")
           
            print(content)
            
        else:
            print(f"\n--- {name} ---")
            print(f"状态: HTTP {r.status_code}")
            print(r.text[:200])
    except Exception as e:
        print(f"\n--- {name} ---")
        print(f"失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
