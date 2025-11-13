# Embedding Service Docker镜像

这是一个基于SigLIP模型的embedding向量生成服务，支持图像和文本的embedding生成。提供了CPU和GPU两个版本的Docker镜像。

## 功能特性

- 🖼️ **图像Embedding**: 支持base64编码、URL或本地路径的图像
- 📝 **文本Embedding**: 支持单条或多条文本的向量化
- 🚀 **高性能**: GPU版本支持CUDA加速
- 💻 **CPU支持**: CPU版本可在无GPU环境下运行
- 🔍 **健康检查**: 内置健康检查接口
- 📦 **Docker化**: 开箱即用的Docker镜像

## API接口

### 1. 图像Embedding
```bash
POST /embed
Content-Type: application/json

{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQ...",  # base64编码
    "https://example.com/image.jpg",        # URL
    "/path/to/image.jpg"                    # 本地路径
  ]
}
```

响应:
```json
[
  [0.123, 0.456, ...],  # 第一张图像的embedding向量
  [0.789, 0.012, ...]   # 第二张图像的embedding向量
]
```

### 2. 文本Embedding
```bash
POST /embed_text
Content-Type: application/json

{
  "texts": ["文本1", "文本2"]  # 或使用 "text": "单个文本"
}
```

响应:
```json
[
  [0.123, 0.456, ...],  # 第一个文本的embedding向量
  [0.789, 0.012, ...]   # 第二个文本的embedding向量
]
```

### 3. 健康检查
```bash
GET /health
```

响应:
```json
{
  "status": "ok",
  "model": "google/siglip2-so400m-patch16-naflex",
  "device": "cuda",
  "cuda_available": true
}
```

## 构建镜像

### CPU版本
```bash
cd /data/embedding-service
docker build -f cpu/Dockerfile -t embedding-service:cpu .
```

### GPU版本
```bash
cd /data/embedding-service
docker build -f gpu/Dockerfile -t embedding-service:gpu .
```

## 运行容器

### CPU版本
```bash
docker run -d \
  --name embedding-service-cpu \
  -p 8080:8080 \
  -e MODEL_NAME=google/siglip2-so400m-patch16-naflex \
  embedding-service:cpu
```

### GPU版本（需要NVIDIA Docker支持）
```bash
docker run -d \
  --name embedding-service-gpu \
  --gpus all \
  -p 8081:8080 \
  -e MODEL_NAME=google/siglip2-so400m-patch16-naflex \
  -e CUDA_VISIBLE_DEVICES=0 \
  embedding-service:gpu
```

## 使用Docker Compose

### 启动CPU版本
```bash
docker-compose up -d embedding-service-cpu
```

### 启动GPU版本
```bash
docker-compose up -d embedding-service-gpu
```

### 同时启动两个版本
```bash
docker-compose up -d
```

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_NAME` | `google/siglip2-so400m-patch16-naflex` | HuggingFace模型名称 |
| `PORT` | `8080` | 服务监听端口 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU版本使用的GPU设备ID |

## 挂载HuggingFace缓存（可选）

为了加速模型加载，可以将HuggingFace缓存目录挂载到容器：

```bash
docker run -d \
  --name embedding-service-gpu \
  --gpus all \
  -p 8081:8080 \
  -v /path/to/huggingface/cache:/app/.cache/huggingface \
  embedding-service:gpu
```

## 测试API

### 测试健康检查
```bash
curl http://localhost:8080/health
```

### 测试文本Embedding
```bash
curl -X POST http://localhost:8080/embed_text \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test embedding"]}'
```

### 测试图像Embedding（使用base64）
```bash
# 首先将图像转换为base64
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)

curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"data:image/jpeg;base64,$IMAGE_B64\"]}"
```

## 性能说明

- **CPU版本**: 适合小规模使用或测试，推理速度较慢
- **GPU版本**: 推荐生产环境使用，推理速度快10-100倍（取决于GPU型号）

## 生产环境部署（推荐）

### 使用Gunicorn生产版本

**构建生产版本镜像**:
```bash
# CPU版本
docker build -f cpu/Dockerfile.prod -t embedding-service:cpu-prod .

# GPU版本
docker build -f gpu/Dockerfile.prod -t embedding-service:gpu-prod .
```

**使用Docker Compose部署生产版本**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**生产版本优势**:
- ✅ 使用Gunicorn WSGI服务器，性能更好
- ✅ 支持多进程/多线程，并发能力提升3-5倍
- ✅ 更好的资源管理和稳定性
- ✅ 支持优雅重启和健康检查

**性能对比**:
- 开发版本（Flask）: 30-50并发，15-25 RPS
- 生产版本（Gunicorn）: 100-200并发，50-100 RPS（CPU）/ 200-400 RPS（GPU）

详细性能分析请参考 [CONCURRENCY_ANALYSIS.md](CONCURRENCY_ANALYSIS.md)

## 性能测试

### 使用benchmark.py工具

```bash
# 安装依赖
pip install aiohttp requests

# 基础测试
python benchmark.py --url http://localhost:8080 --concurrency 10 --requests 100

# 压力测试
python benchmark.py --url http://localhost:8080 --concurrency 50 --requests 500
```

## 注意事项

1. GPU版本需要NVIDIA Docker运行时（nvidia-docker2）
2. 首次运行会下载模型，需要一定时间
3. 模型文件较大（约1-2GB），确保有足够的磁盘空间
4. GPU版本建议至少4GB显存
5. **生产环境请使用生产版本（Gunicorn）**，不要使用Flask开发服务器

## 故障排查

### 检查容器日志
```bash
docker logs embedding-service-gpu
```

### 检查GPU是否可用
```bash
docker exec embedding-service-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### 检查模型加载
查看容器日志中的模型加载信息，确认模型是否正确下载和加载。

## 许可证

本项目使用的模型遵循其原始许可证。请参考HuggingFace上的模型页面了解详情。

