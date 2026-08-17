#!/bin/bash
# 测试 external API 5个接口
# 用法: bash test_api.sh [ts_code] [end_date]
# 示例: bash test_api.sh 300394.SZ 2026-05-21

TS_CODE="${1:-600519.SH}"
END_DATE="${2:-2026-05-21}"
BASE="http://localhost:8888/api/external"

echo "=========================================="
echo " External API 接口测试"
echo " ts_code: $TS_CODE"
echo " end_date: $END_DATE"
echo "=========================================="

# 1. 基本面
echo ""
echo "--- 1. 基本面 /fundamental ---"
curl -s "${BASE}/fundamental/${TS_CODE}" | python3 -m json.tool 2>/dev/null | head -30
echo ""

# 2. 技术面
echo "--- 2. 技术面 /technical ---"
curl -s "${BASE}/technical/${TS_CODE}?end_date=${END_DATE}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
content=d.get('content','')
print(f'状态: OK, content长度: {len(content)}')
print(content[:500])
print('...(省略)...' if len(content)>500 else '')
"

# 3. 博弈面
echo ""
echo "--- 3. 博弈面 /game ---"
curl -s "${BASE}/game/${TS_CODE}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'content: {d.get(\"content\",\"\")[:300]}')
"

# 4. 风险面
echo ""
echo "--- 4. 风险面 /risk ---"
curl -s "${BASE}/risk/${TS_CODE}?end_date=${END_DATE}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
content=d.get('content','')
print(f'状态: OK, content长度: {len(content)}')
print(content[:500])
print('...(省略)...' if len(content)>500 else '')
"

# 5. 新闻舆情
echo ""
echo "--- 5. 新闻舆情 /risk_sentiment ---"
curl -s "${BASE}/risk_sentiment/${TS_CODE}?end_date=${END_DATE}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
content=d.get('content','')
print(f'状态: OK, content长度: {len(content)}')
print(content[:500])
print('...(省略)...' if len(content)>500 else '')
"

# 6. 特殊数据
echo ""
echo "--- 6. 特殊数据 /special_data ---"
curl -s "${BASE}/special_data/${TS_CODE}?end_date=${END_DATE}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
content=d.get('content','')
print(f'状态: OK, content长度: {len(content)}')
print(content[:500])
print('...(省略)...' if len(content)>500 else '')
"

echo ""
echo "=========================================="
echo " 测试完成"
echo "=========================================="
