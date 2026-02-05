# Embedding Service 接口文档

**服务地址**: `http://183.62.232.22:18730`

**服务版本**: 2.0.0

**默认模型**: google/siglip2-so400m-patch16-naflex  
**可用模型**:

| 模型名称 | 向量维度 | 支持输入 | 说明 |
|----------|----------|----------|------|
| google/siglip2-so400m-patch16-naflex | 1152 | 文本 + 图像 | 默认模型，通用多模态 embedding |
| infgrad/stella-mrl-large-zh-v3.5-1792d | 1792 | 仅文本 | 中文文本 embedding，适合语义检索 |
| Marqo/marqo-fashionSigLIP | 768 | 文本 + 图像 | 时尚/服饰领域专用模型，基于 open_clip |

---

## 一、OpenAI 兼容接口（推荐）

### POST /v1/embeddings

生成文本或图像的 embedding 向量，完全兼容 OpenAI Embeddings API 格式。

> 重要提示：
> - 本服务对 OpenAI Embeddings 协议做了扩展：通过 `input_type=image` 支持图像 embedding。
> - 如果你传的是图片 URL / base64 / 本地图片路径，但忘了传 `input_type=image`（默认是 `text`），服务会把它当“普通文本字符串”编码，
>   很容易出现“向量高度相似/疑似塌缩”的错觉（尤其是 URL 前缀相似时）。
> - 你可以通过环境变量开启自动识别/强校验（见下方“环境变量”章节或 README）。

**请求地址**:
```
POST http://183.62.232.22:18730/v1/embeddings
```

**请求头**:
```
Content-Type: application/json
```

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| input | string \| array | ✅ 是 | 输入内容，可以是单个字符串或字符串数组 |
| input_type | string | 否 | 输入类型：`text`（默认）或 `image` |
| model | string | 否 | 模型名称（需在可用模型列表中；未传则用默认模型） |
| encoding_format | string | 否 | 返回格式：`float`（默认）或 `base64` |

**请求示例 - 文本 Embedding**:
```bash
curl -X POST http://183.62.232.22:18730/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一段测试文本",
    "input_type": "text"
  }'
```

**请求示例 - 多文本 Embedding**:
```bash
curl -X POST http://183.62.232.22:18730/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["文本1", "文本2", "文本3"],
    "input_type": "text"
  }'
```

**请求示例 - 图像 Embedding（URL）**:
```bash
curl -X POST http://183.62.232.22:18730/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "https://example.com/image.jpg",
    "input_type": "image"
  }'
```

**请求示例 - 图像 Embedding（Base64）**:
```bash
curl -X POST http://183.62.232.22:18730/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "data:image/jpeg;base64,/9j/4AAQ...",
    "input_type": "image"
  }'
```

**请求示例 - 使用 Marqo 时尚模型**:
```bash
# 文本 Embedding（服饰描述）
curl -X POST http://183.62.232.22:18730/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["red summer dress", "blue running shoes", "white cotton t-shirt"],
    "model": "Marqo/marqo-fashionSigLIP",
    "input_type": "text"
  }'

# 图像 Embedding（服饰图片）
curl -X POST http://183.62.232.22:18730/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "https://example.com/dress.jpg",
    "model": "Marqo/marqo-fashionSigLIP",
    "input_type": "image"
  }'
```

**成功响应** (HTTP 200):
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.123, 0.456, -0.789, ...],
            "index": 0
        }
    ],
    "model": "google/siglip2-so400m-patch16-naflex",
    "usage": {
        "prompt_tokens": 1,
        "total_tokens": 1
    }
}
```

**错误响应** (HTTP 400/500/503):
```json
{
    "error": {
        "message": "错误描述",
        "type": "invalid_request_error",
        "code": "missing_input"
    }
}
```

---

## 二、原有接口

### POST /embed

图像 Embedding 接口

**请求地址**:
```
POST http://183.62.232.22:18730/embed
```

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| images | array | ✅ 是 | 图像列表，支持 URL、Base64、本地路径 |

**请求示例**:
```bash
curl -X POST http://183.62.232.22:18730/embed \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "https://example.com/image1.jpg",
      "data:image/jpeg;base64,/9j/4AAQ..."
    ]
  }'
```

**成功响应** (HTTP 200):
```json
[
    [0.123, 0.456, -0.789, ...],
    [0.321, 0.654, -0.987, ...]
]
```

---

### POST /embed_text

文本 Embedding 接口

**请求地址**:
```
POST http://183.62.232.22:18730/embed_text
```

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| texts | array | 二选一 | 文本列表 |
| text | string | 二选一 | 单个文本 |

**请求示例 - 多文本**:
```bash
curl -X POST http://183.62.232.22:18730/embed_text \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["文本1", "文本2"]
  }'
```

**请求示例 - 单文本**:
```bash
curl -X POST http://183.62.232.22:18730/embed_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段测试文本"
  }'
```

**成功响应** (HTTP 200):
```json
[
    [0.123, 0.456, -0.789, ...],
    [0.321, 0.654, -0.987, ...]
]
```

---

## 三、管理接口

### GET /health

健康检查接口

**请求地址**:
```
GET http://183.62.232.22:18730/health
```

**请求示例**:
```bash
curl http://183.62.232.22:18730/health
```

**成功响应**:
```json
{
    "status": "ok",
    "model": "google/siglip2-so400m-patch16-naflex",
    "available_models": [
        "google/siglip2-so400m-patch16-naflex",
        "infgrad/stella-mrl-large-zh-v3.5-1792d",
        "Marqo/marqo-fashionSigLIP"
    ],
    "device": "cuda",
    "cuda_available": true
}
```

---

### GET /status

服务状态接口（查看并发情况）

**请求地址**:
```
GET http://183.62.232.22:18730/status
```

**请求示例**:
```bash
curl http://183.62.232.22:18730/status
```

**成功响应**:
```json
{
    "max_concurrent": 20,
    "current_processing": 5,
    "current_waiting": 2,
    "max_waiting": 50,
    "available_slots": 15,
    "wait_timeout": 60
}
```

---

### GET /

API 信息接口

**请求地址**:
```
GET http://183.62.232.22:18730/
```

**请求示例**:
```bash
curl http://183.62.232.22:18730/
```

---

## 四、错误码说明

| HTTP 状态码 | 错误类型 | 说明 |
|-------------|----------|------|
| 400 | invalid_request_error | 请求参数错误 |
| 500 | server_error | 服务器内部错误 |
| 503 | rate_limit_error | 服务繁忙，请稍后重试 |
| 504 | timeout_error | 请求在队列中等待超时 |

---

## 五、图像输入格式说明

支持以下三种图像输入格式：

1. **Base64 Data URI**:
   ```
   data:image/jpeg;base64,/9j/4AAQSkZJRg...
   ```

2. **HTTP(S) URL**:
   ```
   https://example.com/path/to/image.jpg
   ```

3. **本地文件路径**（仅限服务器端可访问的路径）:
   ```
   /path/to/local/image.jpg
   ```

---

## 六、Python 调用示例

### 使用 OpenAI SDK（推荐）

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",  # 任意值即可
    base_url="http://183.62.232.22:18730/v1"
)

# 文本 Embedding（默认模型）
response = client.embeddings.create(
    model="infgrad/stella-mrl-large-zh-v3.5-1792d",  # 可选，未传则使用默认模型
    input="这是一段测试文本"
)

embedding = response.data[0].embedding
print(f"向量维度: {len(embedding)}")
```

### 使用 requests

```python
import requests

# 文本 Embedding
response = requests.post(
    "http://183.62.232.22:18730/v1/embeddings",
    json={
        "input": ["文本1", "文本2"],
        "input_type": "text"
    }
)
result = response.json()
embeddings = [item["embedding"] for item in result["data"]]
```

### 使用 Marqo 时尚模型进行服饰检索

```python
import requests
import numpy as np

BASE_URL = "http://183.62.232.22:18730/v1/embeddings"
MODEL = "Marqo/marqo-fashionSigLIP"

def get_text_embedding(texts):
    """获取文本 embedding（服饰描述）"""
    resp = requests.post(BASE_URL, json={
        "input": texts,
        "model": MODEL,
        "input_type": "text"
    })
    return [item["embedding"] for item in resp.json()["data"]]

def get_image_embedding(image_urls):
    """获取图像 embedding（服饰图片）"""
    resp = requests.post(BASE_URL, json={
        "input": image_urls,
        "model": MODEL,
        "input_type": "image"
    })
    return [item["embedding"] for item in resp.json()["data"]]

# 示例：文本搜图片（以文搜图）
query_emb = get_text_embedding(["red summer dress"])[0]
image_embs = get_image_embedding([
    "https://example.com/dress1.jpg",
    "https://example.com/dress2.jpg",
])

# 计算余弦相似度
for i, img_emb in enumerate(image_embs):
    similarity = np.dot(query_emb, img_emb)
    print(f"图片 {i+1} 相似度: {similarity:.4f}")
```

---

## 七、服务限制

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 最大并发数 | 20 | 同时处理的请求数量 |
| 最大等待队列 | 50 | 等待队列中的最大请求数 |
| 等待超时 | 60秒 | 请求在队列中的最大等待时间 |
| 图片下载超时 | 30秒 | 下载远程图片的超时时间 |
| 图片下载重试 | 3次 | 下载失败时的重试次数 |

---

## 八、模型选择建议

| 场景 | 推荐模型 | 说明 |
|------|----------|------|
| 通用图文检索 | google/siglip2-so400m-patch16-naflex | 默认模型，支持多语言，向量维度 1152 |
| 中文文本语义搜索 | infgrad/stella-mrl-large-zh-v3.5-1792d | 中文专精，向量维度 1792，仅支持文本 |
| 时尚/服饰检索 | Marqo/marqo-fashionSigLIP | 服饰领域专用，向量维度 768，支持文本+图片 |

> **注意**：不同模型输出的向量维度不同，不能混用。同一场景应使用同一模型生成的向量进行相似度计算。

---

*文档更新时间: 2026-02-05*
