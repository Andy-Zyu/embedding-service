#!/usr/bin/env python3
"""
通过GitHub API创建仓库并推送代码
"""
import os
import sys
import subprocess
import json
import requests
from getpass import getpass

GITHUB_API = "https://api.github.com"
REPO_NAME = "embedding-service"
OWNER = "Andy-Zyu"

def get_github_token():
    """获取GitHub Token"""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("请输入GitHub Personal Access Token")
        print("如果没有Token，请访问: https://github.com/settings/tokens")
        print("创建新token，权限选择: repo")
        token = getpass("Token: ")
    return token

def create_repo(token, name, description="Embedding向量模型Docker服务，支持CPU/GPU部署", private=False):
    """通过GitHub API创建仓库"""
    url = f"{GITHUB_API}/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": name,
        "description": description,
        "private": private,
        "auto_init": False  # 不初始化README
    }
    
    print(f"正在创建仓库: {OWNER}/{name}...")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        repo_data = response.json()
        print(f"✅ 仓库创建成功!")
        print(f"   地址: {repo_data['html_url']}")
        return True
    elif response.status_code == 422:
        error_data = response.json()
        if "already exists" in str(error_data).lower():
            print(f"⚠️  仓库 {OWNER}/{name} 已存在")
            return True  # 仓库已存在，可以继续推送
        else:
            print(f"❌ 创建失败: {error_data}")
            return False
    else:
        print(f"❌ 创建失败 (HTTP {response.status_code}): {response.text}")
        return False

def check_repo_exists(token, owner, repo):
    """检查仓库是否存在"""
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    return response.status_code == 200

def push_code(repo_path):
    """推送代码到GitHub"""
    print(f"\n正在推送代码到GitHub...")
    try:
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("✅ 代码推送成功!")
            return True
        else:
            print(f"❌ 推送失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 推送出错: {e}")
        return False

def main():
    print("=" * 60)
    print("GitHub仓库创建和代码推送工具")
    print("=" * 60)
    
    # 获取token
    token = get_github_token()
    if not token:
        print("❌ 需要GitHub Token才能继续")
        sys.exit(1)
    
    # 检查仓库是否已存在
    if check_repo_exists(token, OWNER, REPO_NAME):
        print(f"✅ 仓库 {OWNER}/{REPO_NAME} 已存在")
        choice = input("是否继续推送代码? (y/n): ").strip().lower()
        if choice != 'y':
            print("已取消")
            sys.exit(0)
    else:
        # 创建仓库
        private = input("创建私有仓库? (y/n, 默认n): ").strip().lower() == 'y'
        if not create_repo(token, REPO_NAME, private=private):
            print("❌ 仓库创建失败")
            sys.exit(1)
    
    # 推送代码
    repo_path = "/data/embedding-service"
    if push_code(repo_path):
        print("\n" + "=" * 60)
        print("✅ 完成!")
        print(f"   仓库地址: https://github.com/{OWNER}/{REPO_NAME}")
        print("=" * 60)
    else:
        print("\n⚠️  代码推送失败，请手动执行:")
        print(f"   cd {repo_path}")
        print("   git push -u origin main")

if __name__ == "__main__":
    main()

