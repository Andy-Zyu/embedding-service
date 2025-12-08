#!/usr/bin/env python3
"""
Gunicorn配置文件
用于生产环境部署，提供更好的并发性能
"""
import os
import multiprocessing
import torch

# Worker初始化钩子（避免CUDA fork问题）
def post_fork(server, worker):
    """Worker进程fork后的回调
    
    说明：
    - CUDA不支持在fork的子进程中使用，所以不能在主进程预加载CUDA模型
    - 改为在每个worker fork后独立加载模型
    - 虽然每个worker都加载模型，但这是GPU + multiprocessing的限制
    """
    print(f"Worker {worker.pid} forked, preloading model...")
    try:
        from app import load_model
        load_model()
        print(f"Worker {worker.pid}: Model loaded successfully!")
    except Exception as e:
        print(f"Worker {worker.pid}: Failed to load model: {e}")
        import traceback
        traceback.print_exc()

def on_starting(server):
    """Gunicorn启动时的回调"""
    print("=" * 60)
    print("Starting Gunicorn server...")
    print("Model will be loaded in each worker after fork (CUDA requirement)")
    print("=" * 60)

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
preload_app = True  # 预加载应用代码（但不预加载CUDA模型）
worker_tmp_dir = '/dev/shm'  # 使用内存文件系统加速

print(f"Gunicorn configuration:")
print(f"  bind: {bind}")
print(f"  workers: {workers}")
print(f"  worker_class: {worker_class}")
print(f"  threads: {threads}")
print(f"  timeout: {timeout}s")
print(f"  preload_app: {preload_app}")

