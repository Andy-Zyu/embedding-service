# 推送代码到GitHub

## 当前配置

- **远程仓库**: `git@github.com:Andy-Zyu/embedding-service.git`
- **分支**: `main`
- **提交数**: 2个提交
- **文件数**: 21个文件

## 推送步骤

### 1. 确保GitHub仓库已创建

访问 https://github.com/new 创建仓库：
- 仓库名: `embedding-service`
- 所有者: `Andy-Zyu`
- 不要初始化README、.gitignore或license

### 2. 检查SSH密钥

```bash
# 查看SSH公钥
cat ~/.ssh/id_rsa.pub

# 如果输出为空，需要生成SSH密钥
ssh-keygen -t rsa -b 4096 -C "andy-zyu@github.com"

# 将公钥添加到GitHub: https://github.com/settings/keys
```

### 3. 推送代码

```bash
cd /data/embedding-service

# 推送代码
git push -u origin main
```

### 4. 如果SSH有问题，使用HTTPS

```bash
cd /data/embedding-service

# 切换到HTTPS
git remote set-url origin https://github.com/Andy-Zyu/embedding-service.git

# 推送（需要GitHub Personal Access Token）
git push -u origin main
```

## 当前提交

```
7425a40 Add git push instructions
d9c0f7e Initial commit: Embedding Service with CPU/GPU support and production configuration
```

## 包含的文件

- Docker配置文件（CPU/GPU，开发/生产）
- Flask API应用
- Gunicorn生产配置
- 压力测试工具
- 完整文档
- 构建和部署脚本

