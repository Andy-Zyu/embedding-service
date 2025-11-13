# Embedding Service 并发能力快速参考

## 📊 并发能力总结

### 当前版本（Flask开发服务器）

| 版本 | 并发数 | RPS | 适用场景 |
|------|--------|-----|----------|
| **CPU版本** | 30-50 | 15-25 | 开发/测试环境 |
| **GPU版本** | 50-100 | 100-200 | 开发/测试环境 |

### 生产版本（Gunicorn优化）

| 版本 | 并发数 | RPS | 适用场景 |
|------|--------|-----|----------|
| **CPU版本** | 100-200 | 50-100 | 生产环境（4-8核CPU） |
| **GPU版本** | 200-400 | 200-400 | 生产环境（RTX 3090/A100） |

## 🚀 快速部署生产版本

### 1. 构建生产镜像

```bash
cd /data/embedding-service

# CPU版本
docker build -f cpu/Dockerfile.prod -t embedding-service:cpu-prod .

# GPU版本
docker build -f gpu/Dockerfile.prod -t embedding-service:gpu-prod .
```

### 2. 运行生产版本

```bash
# CPU版本（4 workers, 2 threads each）
docker run -d \
  --name embedding-cpu-prod \
  -p 8080:8080 \
  -e WORKERS=4 \
  -e THREADS=2 \
  embedding-service:cpu-prod

# GPU版本（2 workers, 4 threads each）
docker run -d \
  --name embedding-gpu-prod \
  --gpus all \
  -p 8081:8080 \
  -e WORKERS=2 \
  -e THREADS=4 \
  embedding-service:gpu-prod
```

### 3. 使用Docker Compose

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📈 性能测试

### 使用benchmark.py工具

```bash
# 安装依赖
pip install aiohttp requests

# 基础测试
python benchmark.py --url http://localhost:8080 --concurrency 10 --requests 100

# 中等压力测试
python benchmark.py --url http://localhost:8080 --concurrency 50 --requests 500

# 高压力测试
python benchmark.py --url http://localhost:8080 --concurrency 100 --requests 1000
```

## ⚙️ 配置调优

### CPU版本推荐配置

```bash
# 4核CPU
WORKERS=4 THREADS=2

# 8核CPU
WORKERS=8 THREADS=2

# 16核CPU
WORKERS=8 THREADS=4  # 或 WORKERS=16 THREADS=2
```

### GPU版本推荐配置

```bash
# RTX 3090 (24GB)
WORKERS=2 THREADS=4

# A100 (40GB)
WORKERS=2 THREADS=8
```

## 🔧 高并发优化方案

### 方案1: 单实例优化（推荐）

使用Gunicorn生产版本，配置合适的workers和threads。

### 方案2: 多实例部署

```bash
# 启动多个实例
docker run -d -p 8080:8080 --name embedding-1 embedding-service:gpu-prod
docker run -d -p 8081:8080 --name embedding-2 embedding-service:gpu-prod
docker run -d -p 8082:8080 --name embedding-3 embedding-service:gpu-prod

# 使用Nginx负载均衡
# nginx配置示例见下方
```

### Nginx负载均衡配置示例

```nginx
upstream embedding_backend {
    least_conn;
    server localhost:8080;
    server localhost:8081;
    server localhost:8082;
}

server {
    listen 80;
    server_name embedding.example.com;

    location / {
        proxy_pass http://embedding_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📝 性能监控

### 监控指标

- **响应时间**: P50, P95, P99
- **吞吐量**: RPS (Requests Per Second)
- **错误率**: 失败请求百分比
- **资源使用**: CPU, 内存, GPU利用率

### 查看容器资源使用

```bash
# 实时监控
docker stats embedding-service-gpu-prod

# 查看日志
docker logs -f embedding-service-gpu-prod
```

## ⚠️ 注意事项

1. **不要在生产环境使用Flask开发服务器**
2. **GPU版本建议1-2个workers**（避免显存竞争）
3. **CPU版本workers数 = CPU核心数**
4. **监控资源使用，及时调整配置**
5. **高并发场景建议多实例部署**

## 📚 详细文档

- [CONCURRENCY_ANALYSIS.md](CONCURRENCY_ANALYSIS.md) - 详细的性能分析
- [README.md](README.md) - 完整的使用文档

