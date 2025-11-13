# GitHub Personal Access Token 指南

## Token格式

GitHub Personal Access Token (PAT) 的格式通常是：

```
ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

或者（经典版本）：
```
ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**特点**:
- 以 `ghp_` 开头
- 后面跟着一串随机字符（通常是40-50个字符）
- 例如: `ghp_1234567890abcdefghijklmnopqrstuvwxyz123456`

## Token类型

### 1. Fine-grained Personal Access Token (推荐)
- 格式: `github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- 更细粒度的权限控制
- 可以设置过期时间
- 只能访问特定仓库

### 2. Classic Personal Access Token
- 格式: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- 传统格式
- 权限范围更广
- 可以访问所有仓库

## 如何获取Token

### 步骤1: 访问Token设置页面

访问: https://github.com/settings/tokens

或者：
1. 登录GitHub
2. 点击右上角头像
3. 选择 "Settings"
4. 左侧菜单选择 "Developer settings"
5. 选择 "Personal access tokens" -> "Tokens (classic)"

### 步骤2: 生成新Token

**Classic Token（推荐用于脚本）**:
1. 点击 "Generate new token" -> "Generate new token (classic)"
2. 填写信息:
   - **Note**: 描述用途，如 "Embedding Service Repo Creation"
   - **Expiration**: 选择过期时间（建议90天或自定义）
   - **Select scopes**: 勾选 `repo` (全部权限)
     - ✅ repo (全部权限)
     - ✅ workflow (如果需要GitHub Actions)
3. 点击 "Generate token"
4. **重要**: 立即复制token，页面关闭后无法再次查看！

**Fine-grained Token**:
1. 点击 "Generate new token" -> "Generate new token (fine-grained)"
2. 填写信息:
   - **Token name**: 描述用途
   - **Expiration**: 过期时间
   - **Repository access**: 选择 "All repositories" 或特定仓库
   - **Permissions**: 
     - Repository permissions -> Contents: Read and write
     - Repository permissions -> Metadata: Read-only
3. 点击 "Generate token"
4. 复制token

## 使用Token

### 方式1: 环境变量（推荐）

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
./create_repo.sh
```

### 方式2: 直接在命令中（不推荐，不安全）

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ./create_repo.sh
```

### 方式3: 运行时输入（脚本会提示）

```bash
./create_repo.sh
# 脚本会提示输入Token
```

## 安全注意事项

⚠️ **重要**:
1. **不要**将token提交到Git仓库
2. **不要**在公开场合分享token
3. Token一旦泄露，立即撤销并重新生成
4. 使用环境变量存储token，不要硬编码
5. 定期轮换token（更新过期时间）

## 验证Token是否有效

```bash
# 使用curl测试
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user

# 如果返回你的用户信息，说明token有效
```

## 撤销Token

如果token泄露或不再需要：
1. 访问: https://github.com/settings/tokens
2. 找到对应的token
3. 点击 "Revoke" 撤销

## 示例Token（仅用于说明格式，已失效）

```
ghp_1234567890abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz
```

**注意**: 上面的token是示例，不是真实token。

## 获取Token的快速链接

- Classic Token: https://github.com/settings/tokens/new
- Fine-grained Token: https://github.com/settings/tokens/new?type=beta
- 管理现有Token: https://github.com/settings/tokens

