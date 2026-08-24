"""测试 external API 接口（含新增 market_environment / schema）"""
import requests
import sys
import json

TS_CODE = sys.argv[1] if len(sys.argv) > 1 else "300438.SZ"
END_DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-05-21"
BASE = "http://localhost:8000/api/external"

endpoints = [
    ("基本面 /fundamental", f"{BASE}/fundamental/{TS_CODE}", {}),
    ("技术面 /technical", f"{BASE}/technical/{TS_CODE}", {"end_date": END_DATE}),
    ("博弈面 /game", f"{BASE}/game/{TS_CODE}", {}),
    ("市场宏观环境 /market_environment", f"{BASE}/market_environment", {"end_date": END_DATE}),
    ("字段含义 /schema/technical", f"{BASE}/schema/technical", {}),
    ("新闻舆情 /risk_sentiment", f"{BASE}/risk_sentiment/{TS_CODE}", {"end_date": END_DATE}),
]

print(f"External API 测试 | ts_code={TS_CODE} end_date={END_DATE}")
print("=" * 60)

for name, url, params in endpoints:
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            d = r.json()
            text = d.get("content") or json.dumps(d, ensure_ascii=False, indent=1)
            print(f"\n--- {name} ---")
            print(f"状态: OK, 长度: {len(text)}, keys: {list(d.keys())}")
            print(text[:1200])
        else:
            print(f"\n--- {name} ---")
            print(f"状态: HTTP {r.status_code}")
            print(r.text[:200])
    except Exception as e:
        print(f"\n--- {name} ---")
        print(f"失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
