#!/usr/bin/env python3
"""
Embedding Service 并发压力测试工具
测试服务在不同并发量下的性能表现
"""
import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor
import requests

class BenchmarkResult:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.errors = []
        
    def add_result(self, success: bool, response_time: float, error: str = None):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.response_times.append(response_time)
        else:
            self.failed_requests += 1
            if error:
                self.errors.append(error)
    
    def get_stats(self) -> Dict:
        if not self.response_times:
            return {
                'total': self.total_requests,
                'success': self.successful_requests,
                'failed': self.failed_requests,
                'success_rate': 0.0,
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'median_time': 0.0,
                'p95_time': 0.0,
                'p99_time': 0.0,
                'rps': 0.0,
                'errors': self.errors[:10]  # 只显示前10个错误
            }
        
        return {
            'total': self.total_requests,
            'success': self.successful_requests,
            'failed': self.failed_requests,
            'success_rate': self.successful_requests / self.total_requests * 100,
            'avg_time': statistics.mean(self.response_times),
            'min_time': min(self.response_times),
            'max_time': max(self.response_times),
            'median_time': statistics.median(self.response_times),
            'p95_time': self._percentile(self.response_times, 95),
            'p99_time': self._percentile(self.response_times, 99),
            'rps': self.successful_requests / (sum(self.response_times) / len(self.response_times)) if self.response_times else 0,
            'errors': self.errors[:10]
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

def test_text_embedding_sync(url: str, text: str, timeout: int = 30) -> tuple:
    """同步测试文本embedding"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{url}/embed_text",
            json={"texts": [text]},
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            return True, elapsed, None
        else:
            return False, elapsed, f"HTTP {response.status_code}: {response.text[:100]}"
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)

async def test_text_embedding_async(session: aiohttp.ClientSession, url: str, text: str, timeout: int = 30) -> tuple:
    """异步测试文本embedding"""
    start_time = time.time()
    try:
        async with session.post(
            f"{url}/embed_text",
            json={"texts": [text]},
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            elapsed = time.time() - start_time
            if response.status == 200:
                await response.json()  # 读取响应
                return True, elapsed, None
            else:
                text_content = await response.text()
                return False, elapsed, f"HTTP {response.status}: {text_content[:100]}"
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)

def run_sync_benchmark(url: str, concurrency: int, total_requests: int, text: str = "Hello world"):
    """同步并发测试"""
    print(f"\n开始同步测试: 并发数={concurrency}, 总请求数={total_requests}")
    
    result = BenchmarkResult()
    start_time = time.time()
    
    def worker():
        for _ in range(total_requests // concurrency):
            success, elapsed, error = test_text_embedding_sync(url, text)
            result.add_result(success, elapsed, error)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    
    total_time = time.time() - start_time
    stats = result.get_stats()
    stats['total_time'] = total_time
    stats['actual_rps'] = stats['success'] / total_time if total_time > 0 else 0
    
    return stats

async def run_async_benchmark(url: str, concurrency: int, total_requests: int, text: str = "Hello world"):
    """异步并发测试"""
    print(f"\n开始异步测试: 并发数={concurrency}, 总请求数={total_requests}")
    
    result = BenchmarkResult()
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def worker():
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                success, elapsed, error = await test_text_embedding_async(session, url, text)
                result.add_result(success, elapsed, error)
    
    tasks = [worker() for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    stats = result.get_stats()
    stats['total_time'] = total_time
    stats['actual_rps'] = stats['success'] / total_time if total_time > 0 else 0
    
    return stats

def print_stats(stats: Dict, title: str = "测试结果"):
    """打印统计结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"总请求数:      {stats['total']}")
    print(f"成功请求:      {stats['success']}")
    print(f"失败请求:      {stats['failed']}")
    print(f"成功率:        {stats['success_rate']:.2f}%")
    print(f"\n响应时间统计 (秒):")
    print(f"  平均:        {stats['avg_time']:.3f}s")
    print(f"  中位数:      {stats['median_time']:.3f}s")
    print(f"  最小值:      {stats['min_time']:.3f}s")
    print(f"  最大值:      {stats['max_time']:.3f}s")
    print(f"  P95:         {stats['p95_time']:.3f}s")
    print(f"  P99:         {stats['p99_time']:.3f}s")
    print(f"\n吞吐量:")
    print(f"  理论RPS:     {stats['rps']:.2f} req/s")
    print(f"  实际RPS:     {stats['actual_rps']:.2f} req/s")
    print(f"  总耗时:      {stats['total_time']:.2f}s")
    
    if stats['errors']:
        print(f"\n错误信息 (前10个):")
        for i, error in enumerate(stats['errors'], 1):
            print(f"  {i}. {error[:100]}")

def main():
    parser = argparse.ArgumentParser(description='Embedding Service 压力测试工具')
    parser.add_argument('--url', default='http://localhost:8080', help='服务URL')
    parser.add_argument('--concurrency', type=int, default=10, help='并发数')
    parser.add_argument('--requests', type=int, default=100, help='总请求数')
    parser.add_argument('--mode', choices=['sync', 'async', 'both'], default='both', help='测试模式')
    parser.add_argument('--text', default='Hello world', help='测试文本')
    parser.add_argument('--timeout', type=int, default=30, help='请求超时时间(秒)')
    
    args = parser.parse_args()
    
    print(f"测试配置:")
    print(f"  服务URL:     {args.url}")
    print(f"  并发数:      {args.concurrency}")
    print(f"  总请求数:    {args.requests}")
    print(f"  测试模式:    {args.mode}")
    print(f"  测试文本:    {args.text}")
    
    # 先测试健康检查
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"\n服务状态: 正常")
            print(f"  模型:     {health_data.get('model', 'N/A')}")
            print(f"  设备:     {health_data.get('device', 'N/A')}")
            print(f"  CUDA:     {health_data.get('cuda_available', False)}")
        else:
            print(f"\n警告: 健康检查失败 (HTTP {response.status_code})")
    except Exception as e:
        print(f"\n错误: 无法连接到服务 - {e}")
        return
    
    # 运行测试
    if args.mode in ['sync', 'both']:
        stats = run_sync_benchmark(args.url, args.concurrency, args.requests, args.text)
        print_stats(stats, "同步测试结果")
    
    if args.mode in ['async', 'both']:
        stats = asyncio.run(run_async_benchmark(args.url, args.concurrency, args.requests, args.text))
        print_stats(stats, "异步测试结果")

if __name__ == '__main__':
    main()

