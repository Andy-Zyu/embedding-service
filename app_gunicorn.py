#!/usr/bin/env python3
"""
Gunicorn配置文件
用于生产环境部署，提供更好的并发性能
"""
import os
import multiprocessing
import torch

# 预加载模型（在worker启动前）
def on_starting(server):
    """Gunicorn启动时的回调"""
    print("Preloading model...")
    from app import load_model
    load_model()
    print("Model preloaded successfully")

# Gunicorn配置
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8080')}"

# Worker配置
workers = int(os.getenv('WORKERS', '0'))  # 0表示自动检测CPU核心数
if workers == 0:
    workers = multiprocessing.cpu_count()
    # GPU版本建议使用更少的workers（避免GPU内存竞争）
    if torch.cuda.is_available():
        workers = min(workers, 2)  # GPU版本建议1-2个worker
        print(f"GPU detected, setting workers to {workers}")
    else:
        print(f"CPU detected, setting workers to {workers}")

worker_class = os.getenv('WORKER_CLASS', 'sync')  # sync, gevent, gthread
threads = int(os.getenv('THREADS', '2'))  # 每个worker的线程数
timeout = int(os.getenv('TIMEOUT', '120'))  # 请求超时时间
keepalive = int(os.getenv('KEEPALIVE', '5'))  # Keep-alive连接时间
max_requests = int(os.getenv('MAX_REQUESTS', '1000'))  # 每个worker处理的最大请求数
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', '100'))  # 随机抖动

# 日志配置
accesslog = '-'  # 输出到stdout
errorlog = '-'   # 输出到stderr
loglevel = os.getenv('LOG_LEVEL', 'info')

# 性能优化
preload_app = True  # 预加载应用，共享模型内存
worker_tmp_dir = '/dev/shm'  # 使用内存文件系统加速

print(f"Gunicorn configuration:")
print(f"  bind: {bind}")
print(f"  workers: {workers}")
print(f"  worker_class: {worker_class}")
print(f"  threads: {threads}")
print(f"  timeout: {timeout}s")
print(f"  preload_app: {preload_app}")

