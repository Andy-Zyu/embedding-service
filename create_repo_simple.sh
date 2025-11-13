#!/bin/bash
# 简单的GitHub仓库创建脚本（使用GitHub API）

set -e

REPO_NAME="embedding-service"
OWNER="Andy-Zyu"
GITHUB_API="https://api.github.com"

echo "=========================================="
echo "GitHub仓库创建工具"
echo "=========================================="

# 检查是否提供了token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "请设置GitHub Token:"
    echo "  export GITHUB_TOKEN=your_token_here"
    echo ""
    echo "或者直接运行:"
    echo "  GITHUB_TOKEN=your_token bash create_repo_simple.sh"
    echo ""
    read -sp "请输入GitHub Token: " GITHUB_TOKEN
    echo ""
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ 需要GitHub Token"
    exit 1
fi

# 创建仓库
echo "正在创建仓库: $OWNER/$REPO_NAME..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    -d "{\"name\":\"$REPO_NAME\",\"description\":\"Embedding向量模型Docker服务，支持CPU/GPU部署\",\"private\":false,\"auto_init\":false}" \
    "$GITHUB_API/user/repos")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "201" ]; then
    echo "✅ 仓库创建成功!"
    REPO_URL=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin)['html_url'])" 2>/dev/null || echo "https://github.com/$OWNER/$REPO_NAME")
    echo "   地址: $REPO_URL"
    
    # 推送代码
    echo ""
    echo "正在推送代码..."
    cd /data/embedding-service
    git push -u origin main && echo "✅ 代码推送成功!" || echo "⚠️  推送失败，请手动执行: git push -u origin main"
    
elif [ "$HTTP_CODE" = "422" ]; then
    if echo "$BODY" | grep -q "already exists"; then
        echo "⚠️  仓库已存在，直接推送代码..."
        cd /data/embedding-service
        git push -u origin main && echo "✅ 代码推送成功!" || echo "⚠️  推送失败，请手动执行: git push -u origin main"
    else
        echo "❌ 创建失败: $BODY"
        exit 1
    fi
else
    echo "❌ 创建失败 (HTTP $HTTP_CODE): $BODY"
    exit 1
fi

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="

