#!/bin/bash
# 通过GitHub API创建仓库并推送代码

set -e

REPO_NAME="embedding-service"
OWNER="Andy-Zyu"
GITHUB_API="https://api.github.com"
REPO_PATH="/data/embedding-service"

echo "=========================================="
echo "GitHub仓库创建和推送工具"
echo "=========================================="

# 获取GitHub Token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "请输入GitHub Personal Access Token"
    echo "如果没有Token，请访问: https://github.com/settings/tokens"
    echo "创建新token，权限选择: repo"
    read -sp "Token: " GITHUB_TOKEN
    echo ""
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ 需要GitHub Token才能继续"
    exit 1
fi

# 检查仓库是否已存在
echo "检查仓库是否存在..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "$GITHUB_API/repos/$OWNER/$REPO_NAME")

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ 仓库 $OWNER/$REPO_NAME 已存在"
    SKIP_CREATE=true
elif [ "$HTTP_CODE" = "404" ]; then
    echo "仓库不存在，准备创建..."
    SKIP_CREATE=false
else
    echo "❌ 检查仓库时出错 (HTTP $HTTP_CODE)"
    exit 1
fi

# 创建仓库（如果不存在）
if [ "$SKIP_CREATE" = "false" ]; then
    echo "正在创建仓库: $OWNER/$REPO_NAME..."
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"$REPO_NAME\",
            \"description\": \"Embedding向量模型Docker服务，支持CPU/GPU部署\",
            \"private\": false,
            \"auto_init\": false
        }" \
        "$GITHUB_API/user/repos")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" = "201" ]; then
        echo "✅ 仓库创建成功!"
        REPO_URL=$(echo "$BODY" | grep -o '"html_url":"[^"]*' | cut -d'"' -f4 || echo "https://github.com/$OWNER/$REPO_NAME")
        echo "   地址: $REPO_URL"
    elif [ "$HTTP_CODE" = "422" ]; then
        if echo "$BODY" | grep -qi "already exists"; then
            echo "⚠️  仓库已存在（可能刚创建）"
        else
            echo "❌ 创建失败: $BODY"
            exit 1
        fi
    else
        echo "❌ 创建失败 (HTTP $HTTP_CODE): $BODY"
        exit 1
    fi
fi

# 推送代码
echo ""
echo "正在推送代码到GitHub..."
cd "$REPO_PATH"

# 确保远程仓库URL正确
git remote set-url origin "git@github.com:$OWNER/$REPO_NAME.git" 2>/dev/null || true

# 推送
if git push -u origin main 2>&1; then
    echo "✅ 代码推送成功!"
    echo ""
    echo "=========================================="
    echo "完成!"
    echo "=========================================="
    echo "仓库地址: https://github.com/$OWNER/$REPO_NAME"
else
    echo "⚠️  推送失败，可能的原因:"
    echo "   1. SSH密钥未添加到GitHub"
    echo "   2. 仓库权限问题"
    echo ""
    echo "可以尝试使用HTTPS方式:"
    echo "   git remote set-url origin https://github.com/$OWNER/$REPO_NAME.git"
    echo "   git push -u origin main"
    exit 1
fi

