#!/bin/bash
# API测试脚本

API_URL="${1:-http://localhost:8080}"

echo "Testing Embedding Service API at: $API_URL"
echo "=========================================="

# 测试健康检查
echo ""
echo "1. Testing /health endpoint..."
curl -s "$API_URL/health" | python3 -m json.tool
if [ $? -eq 0 ]; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
fi

# 测试文本embedding
echo ""
echo "2. Testing /embed_text endpoint..."
RESPONSE=$(curl -s -X POST "$API_URL/embed_text" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test embedding"]}')

if echo "$RESPONSE" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
    echo "✓ Text embedding test passed"
    echo "Response preview:"
    echo "$RESPONSE" | python3 -m json.tool | head -20
else
    echo "✗ Text embedding test failed"
    echo "Response: $RESPONSE"
fi

# 测试根路径
echo ""
echo "3. Testing / endpoint..."
curl -s "$API_URL/" | python3 -m json.tool

echo ""
echo "=========================================="
echo "Testing completed!"

