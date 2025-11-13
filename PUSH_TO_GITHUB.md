# 推送到GitHub仓库

## 当前状态

✅ Git仓库已初始化
✅ 所有文件已提交到本地仓库
✅ 分支: main

## 添加远程仓库并推送

### 方法1: 推送到新仓库

如果你还没有创建GitHub仓库，请先到GitHub创建新仓库，然后执行：

```bash
cd /data/embedding-service

# 添加远程仓库（替换为你的仓库地址）
git remote add origin git@github.com:li-yongyu/embedding-service.git

# 或者使用HTTPS
# git remote add origin https://github.com/li-yongyu/embedding-service.git

# 推送到远程仓库
git push -u origin main
```

### 方法2: 推送到已存在的仓库

如果仓库已存在：

```bash
cd /data/embedding-service

# 添加远程仓库
git remote add origin <你的仓库地址>

# 推送
git push -u origin main
```

### 方法3: 使用GitHub CLI (如果已安装)

```bash
cd /data/embedding-service

# 创建并推送（会自动创建仓库）
gh repo create embedding-service --public --source=. --remote=origin --push
```

## 查看当前状态

```bash
# 查看远程仓库
git remote -v

# 查看提交历史
git log --oneline

# 查看文件状态
git status
```

## 后续更新

```bash
# 添加更改
git add .

# 提交
git commit -m "描述你的更改"

# 推送
git push
```

