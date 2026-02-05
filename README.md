# Embedding Service Docker镜像

这是一个基于SigLIP模型的embedding向量生成服务，支持图像和文本的embedding生成。提供了CPU和GPU两个版本的Docker镜像，并支持按请求选择不同模型。

## 功能特性

- 🖼️ **图像Embedding**: 支持base64编码、URL或本地路径的图像
- 📝 **文本Embedding**: 支持单条或多条文本的向量化
- 🚀 **高性能**: GPU版本支持CUDA加速
- 💻 **CPU支持**: CPU版本可在无GPU环境下运行
- 🔍 **健康检查**: 内置健康检查接口
- 🧩 **多模型选择**: 可配置可用模型列表，请求中指定 `model` 进行选择
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

### 配置国内镜像源（推荐）

为了加速构建，已内置以下国内镜像源配置：
- ✅ **pip镜像源**: 清华大学镜像（`pypi.tuna.tsinghua.edu.cn`）
- ✅ **HuggingFace镜像**: `hf-mirror.com`

**Docker基础镜像加速**（可选）：
如需加速Docker基础镜像拉取，请配置Docker镜像加速器，详见 [DOCKER_MIRROR_SETUP.md](DOCKER_MIRROR_SETUP.md)

### CPU版本
```bash
cd /data/embedding-service
docker build -f cpu/Dockerfile -t embedding-service:cpu .
```

### GPU版本
```bash
cd /data/embedding-service
docker build -f gpu/Dockerfile -t embedding-service:gpu .

# 如果在Apple Silicon Mac上构建（会显示平台警告，但可以正常构建）
# Docker会自动处理平台转换，构建的镜像可以在Linux服务器上使用
```

## 运行容器

### CPU版本
```bash
docker run -d \
  --name embedding-service-cpu \
  -p 8080:8080 \
  -e DEFAULT_MODEL_NAME=google/siglip2-so400m-patch16-naflex \
  -e AVAILABLE_MODELS=google/siglip2-so400m-patch16-naflex,infgrad/stella-mrl-large-zh-v3.5-1792d \
  -e SENTENCE_TRANSFORMERS_MODELS=infgrad/stella-mrl-large-zh-v3.5-1792d \
  embedding-service:cpu
```

### GPU版本（需要NVIDIA Docker支持）
```bash
docker run -d \
  --name embedding-service-gpu \
  --gpus all \
  -p 8081:8080 \
  -e DEFAULT_MODEL_NAME=google/siglip2-so400m-patch16-naflex \
  -e AVAILABLE_MODELS=google/siglip2-so400m-patch16-naflex,infgrad/stella-mrl-large-zh-v3.5-1792d \
  -e SENTENCE_TRANSFORMERS_MODELS=infgrad/stella-mrl-large-zh-v3.5-1792d \
  -e CUDA_VISIBLE_DEVICES=0 \
  embedding-service:gpu
```

## 使用Docker Compose

### 单实例部署（适合中小规模）

```bash
# 启动CPU版本
docker compose up -d embedding-service-cpu

# 启动GPU版本
docker compose up -d embedding-service-gpu

# 同时启动两个版本
docker compose up -d
```

### 多实例横向扩展（适合大规模并发）

**方案1: 使用Docker Compose Scale（推荐）**

```bash
# 启动3个CPU实例 + 2个GPU实例
docker compose -f docker-compose.scale.yml up -d \
  --scale embedding-service-cpu=3 \
  --scale embedding-service-gpu=2

# 查看运行状态
docker compose -f docker-compose.scale.yml ps
```

**方案2: 使用Nginx负载均衡（统一入口）**

```bash
# 启动多实例 + Nginx负载均衡器
docker compose -f docker-compose.scale.yml up -d \
  --scale embedding-service-cpu=3 \
  --scale embedding-service-gpu=2

# 通过Nginx访问（端口80）
curl http://localhost/health
```

## 并发性能策略

### Docker内部并发（单实例优化）

每个Docker容器内部使用 **Gunicorn** 进行并发优化：

- **CPU版本**: 4 workers × 2 threads = **8并发**
- **GPU版本**: 2 workers × 4 threads = **8并发**

**适用场景**: 中小规模并发（100-200 RPS）

### Docker Compose横向扩展（大规模并发）

通过启动多个Docker实例实现横向扩展：

- **3个CPU实例**: 3 × 8并发 = **24并发**，约 **150-300 RPS**
- **5个GPU实例**: 5 × 8并发 = **40并发**，约 **1000-2000 RPS**

**适用场景**: 大规模并发（1000+ RPS）

### 推荐配置

| 并发需求 | CPU实例数 | GPU实例数 | 预期RPS |
|---------|----------|----------|---------|
| 小规模（<100） | 1 | 1 | 50-200 |
| 中规模（100-500） | 2-3 | 1-2 | 200-1000 |
| 大规模（500-2000） | 5-10 | 3-5 | 1000-5000 |
| 超大规模（2000+） | 10+ | 5+ | 5000+ |

**注意**: 
- Docker内部并发（Gunicorn）是**单实例优化**，受限于单机资源
- Docker Compose横向扩展是**多实例扩展**，可以突破单机限制
- **推荐**: 先用Docker内部并发，需要更高并发时再横向扩展

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEFAULT_MODEL_NAME` | `google/siglip2-so400m-patch16-naflex` | 默认模型名称 |
| `AVAILABLE_MODELS` | `google/...` | 可用模型列表（逗号分隔） |
| `PRELOAD_MODELS` | `0` | 是否启动时预加载全部模型 |
| `SENTENCE_TRANSFORMERS_MODELS` | `infgrad/...` | 使用 SentenceTransformers 加载的模型列表（逗号分隔） |
| `PORT` | `8080` | 服务监听端口 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `WORKERS` | `4` (CPU) / `2` (GPU) | Gunicorn worker进程数 |
| `THREADS` | `2` | 每个worker的线程数 |
| `WORKER_CLASS` | `sync` | Worker类型（sync/gevent/gthread） |
| `TIMEOUT` | `120` | 请求超时时间（秒） |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU版本使用的GPU设备ID |
| `AUTO_DETECT_INPUT_TYPE` | `0` | 仅对 `/v1/embeddings` 生效：当请求未显式提供 `input_type` 时，若输入**整体看起来像图片**（data:image/、图片URL、图片路径），自动按 `image` 处理 |
| `REJECT_MISMATCH_INPUT_TYPE` | `0` | 仅对 `/v1/embeddings` 生效：当 `input_type=text` 但输入看起来像图片时直接返回 400，避免把图片URL/base64当文本导致“疑似向量塌缩” |

## 挂载HuggingFace缓存（可选）

为了加速模型加载，可以将HuggingFace缓存目录挂载到容器（推荐使用项目内 `./hf_cache`）：

```bash
docker run -d \
  --name embedding-service-gpu \
  --gpus all \
  -p 8081:8080 \
  -v ./hf_cache:/app/.cache/huggingface \
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

**所有版本默认使用Gunicorn生产配置**，已优化并发性能。

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

