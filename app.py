#!/usr/bin/env python3
"""
Embedding Service API
支持图像和文本的embedding向量生成服务
兼容 OpenAI Embeddings API 格式
"""
import io
import base64
import os
import time
import threading
from functools import wraps
from typing import List, Union, Optional
from urllib.parse import urlparse

import requests
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# 模型配置 - 可以通过环境变量覆盖
MODEL_NAME = os.getenv('MODEL_NAME', 'google/siglip2-so400m-patch16-naflex')
PORT = int(os.getenv('PORT', '8080'))
HOST = os.getenv('HOST', '0.0.0.0')

# ============ 并发控制配置 ============
MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT', '20'))   # 单实例最大并发
MAX_WAITING = int(os.getenv('MAX_WAITING', '50'))         # 最大等待队列长度
WAIT_TIMEOUT = float(os.getenv('WAIT_TIMEOUT', '60'))     # 等待超时时间(秒)

# ============ 图片下载配置 ============
IMAGE_DOWNLOAD_RETRIES = int(os.getenv('IMAGE_DOWNLOAD_RETRIES', '3'))    # 下载重试次数
IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv('IMAGE_DOWNLOAD_TIMEOUT', '30'))   # 下载超时时间(秒)
IMAGE_DOWNLOAD_RETRY_DELAY = float(os.getenv('IMAGE_DOWNLOAD_RETRY_DELAY', '1.0'))  # 重试间隔(秒)

# 信号量和计数器
_semaphore = threading.Semaphore(MAX_CONCURRENT)
_waiting_count = 0
_waiting_lock = threading.Lock()

# 全局变量
model = None
processor = None


def load_model():
    """加载模型和处理器"""
    global model, processor
    
    if model is None:
        print(f"Loading model: {MODEL_NAME}")
        device_map = 'auto' if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"Model loaded successfully on device: {next(model.parameters()).device}")
    
    return model, processor


# ============ 限流装饰器 ============
def rate_limited(f):
    """限流装饰器：控制并发数量，超过限制时快速失败"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global _waiting_count
        
        # 检查等待队列是否已满
        with _waiting_lock:
            if _waiting_count >= MAX_WAITING:
                return jsonify({
                    'error': {
                        'message': 'Service busy, please retry later',
                        'type': 'rate_limit_error',
                        'code': 'rate_limited'
                    },
                    'current_waiting': _waiting_count,
                    'max_waiting': MAX_WAITING
                }), 503
            _waiting_count += 1
        
        try:
            # 尝试获取信号量，带超时
            acquired = _semaphore.acquire(timeout=WAIT_TIMEOUT)
            if not acquired:
                return jsonify({
                    'error': {
                        'message': 'Request timeout while waiting in queue',
                        'type': 'timeout_error',
                        'code': 'timeout'
                    },
                    'timeout': WAIT_TIMEOUT
                }), 504
            
            try:
                return f(*args, **kwargs)
            finally:
                _semaphore.release()
        finally:
            with _waiting_lock:
                _waiting_count -= 1
    
    return decorated_function


# ============ 图片下载重试机制 ============
def download_image_with_retry(
    url: str,
    retries: int = None,
    timeout: int = None,
    retry_delay: float = None
) -> Image.Image:
    """
    下载图片，支持重试机制
    
    Args:
        url: 图片URL
        retries: 重试次数
        timeout: 超时时间(秒)
        retry_delay: 重试间隔(秒)
    
    Returns:
        PIL.Image.Image
    
    Raises:
        Exception: 下载失败
    """
    retries = retries or IMAGE_DOWNLOAD_RETRIES
    timeout = timeout or IMAGE_DOWNLOAD_TIMEOUT
    retry_delay = retry_delay or IMAGE_DOWNLOAD_RETRY_DELAY
    
    last_error = None
    
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; EmbeddingService/1.0)'
                }
            )
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            return img.convert('RGB')
            
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout downloading image (attempt {attempt + 1}/{retries}): {url}"
            print(f"[WARN] {last_error}")
        except requests.exceptions.RequestException as e:
            last_error = f"Failed to download image (attempt {attempt + 1}/{retries}): {url}, error: {str(e)}"
            print(f"[WARN] {last_error}")
        except Exception as e:
            last_error = f"Failed to process image (attempt {attempt + 1}/{retries}): {url}, error: {str(e)}"
            print(f"[WARN] {last_error}")
        
        # 如果不是最后一次尝试，等待后重试
        if attempt < retries - 1:
            time.sleep(retry_delay * (attempt + 1))  # 指数退避
    
    raise Exception(f"Failed to download image after {retries} attempts: {last_error}")


def load_image_any(x: Union[str, bytes]) -> Image.Image:
    """
    加载图像，支持多种格式：
    - base64 data URI (data:image/...)
    - HTTP(S) URL (带重试)
    - 本地文件路径
    
    Args:
        x: 图像源
    
    Returns:
        PIL.Image.Image
    """
    if isinstance(x, str):
        # Base64 data URI
        if x.startswith('data:'):
            try:
                b64 = x.split(',', 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {str(e)}")
        
        # HTTP(S) URL - 使用重试机制
        if x.startswith(('http://', 'https://')):
            return download_image_with_retry(x)
        
        # 本地文件路径
        try:
            img = load_image(x)
            return img.convert('RGB') if hasattr(img, 'mode') and img.mode != 'RGB' else img
        except Exception as e:
            raise ValueError(f"Failed to load image from path: {x}, error: {str(e)}")
    
    raise ValueError(f"Unsupported image input type: {type(x)}")


def encode_texts(texts: Union[str, List[str]]) -> List[List[float]]:
    """编码文本为向量"""
    if isinstance(texts, str):
        texts = [texts]
    
    model, processor = load_model()
    inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    
    return feats.detach().cpu().float().tolist()


def encode_images(images: List[Union[str, bytes]]) -> List[List[float]]:
    """编码图像为向量"""
    model, processor = load_model()
    
    # 加载图像（带重试机制）
    pil_images = []
    for idx, img in enumerate(images):
        try:
            pil_images.append(load_image_any(img))
        except Exception as e:
            raise ValueError(f"Failed to load image at index {idx}: {str(e)}")
    
    inputs = processor(images=pil_images, return_tensors='pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    
    return feats.detach().cpu().float().tolist()


# Flask应用
app = Flask(__name__)


# ============ OpenAI 兼容接口 ============
@app.route('/v1/embeddings', methods=['POST'])
@rate_limited
def openai_embeddings():
    """
    OpenAI 兼容的 Embeddings API
    
    支持文本和图像输入，返回标准 OpenAI 格式
    
    请求格式:
    {
        "input": "text" 或 ["text1", "text2"] 或 图片URL/base64,
        "model": "可选，会被忽略",
        "encoding_format": "float" 或 "base64" (默认 float),
        "input_type": "text" 或 "image" (默认 text)
    }
    
    响应格式 (OpenAI 标准):
    {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [...],
                "index": 0
            }
        ],
        "model": "模型名称",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
    """
    try:
        data = request.get_json(force=True)
        
        # 解析输入
        input_data = data.get('input')
        if input_data is None:
            return jsonify({
                'error': {
                    'message': 'Missing required field: input',
                    'type': 'invalid_request_error',
                    'code': 'missing_input'
                }
            }), 400
        
        # 输入类型：text 或 image
        input_type = data.get('input_type', 'text')
        encoding_format = data.get('encoding_format', 'float')
        
        # 转换为列表
        if isinstance(input_data, str):
            inputs = [input_data]
        elif isinstance(input_data, list):
            inputs = input_data
        else:
            return jsonify({
                'error': {
                    'message': 'input must be a string or array of strings',
                    'type': 'invalid_request_error',
                    'code': 'invalid_input'
                }
            }), 400
        
        # 根据类型编码
        if input_type == 'image':
            embeddings = encode_images(inputs)
        else:
            embeddings = encode_texts(inputs)
        
        # 构建响应数据
        response_data = []
        for idx, emb in enumerate(embeddings):
            embedding_value = emb
            
            # 如果需要 base64 编码
            if encoding_format == 'base64':
                import struct
                binary = struct.pack(f'{len(emb)}f', *emb)
                embedding_value = base64.b64encode(binary).decode('utf-8')
            
            response_data.append({
                'object': 'embedding',
                'embedding': embedding_value,
                'index': idx
            })
        
        return jsonify({
            'object': 'list',
            'data': response_data,
            'model': MODEL_NAME,
            'usage': {
                'prompt_tokens': len(inputs),  # 简化处理
                'total_tokens': len(inputs)
            }
        })
        
    except ValueError as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'invalid_request_error',
                'code': 'invalid_input'
            }
        }), 400
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'server_error',
                'code': 'internal_error'
            }
        }), 500


# ============ 原有接口（保留兼容性）============
@app.route('/embed', methods=['POST'])
@rate_limited
def embed():
    """图像embedding接口（原有接口）"""
    try:
        data = request.get_json(force=True)
        if 'images' not in data:
            return jsonify({'error': 'missing images field'}), 400
        
        images = data['images']
        if not isinstance(images, list):
            return jsonify({'error': 'images must be a list'}), 400
        
        embeddings = encode_images(images)
        return jsonify(embeddings)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/embed_text', methods=['POST'])
@rate_limited
def embed_text():
    """文本embedding接口（原有接口）"""
    try:
        data = request.get_json(force=True)
        texts = data.get('texts') or data.get('text')
        
        if texts is None:
            return jsonify({'error': 'missing texts or text field'}), 400
        
        embeddings = encode_texts(texts)
        return jsonify(embeddings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 管理接口 ============
@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available()
    })


@app.route('/status', methods=['GET'])
def status():
    """服务状态接口 - 查看当前并发情况"""
    current_processing = MAX_CONCURRENT - _semaphore._value
    return jsonify({
        'max_concurrent': MAX_CONCURRENT,
        'current_processing': current_processing,
        'current_waiting': _waiting_count,
        'max_waiting': MAX_WAITING,
        'available_slots': MAX_CONCURRENT - current_processing,
        'wait_timeout': WAIT_TIMEOUT
    })


@app.route('/', methods=['GET'])
def index():
    """API信息"""
    return jsonify({
        'service': 'Embedding Service',
        'version': '2.0.0',
        'model': MODEL_NAME,
        'endpoints': {
            '/v1/embeddings': {
                'method': 'POST',
                'description': 'OpenAI 兼容的 Embeddings API',
                'body': {
                    'input': 'string 或 array - 文本或图片URL/base64',
                    'input_type': 'text 或 image (默认 text)',
                    'encoding_format': 'float 或 base64 (默认 float)'
                }
            },
            '/embed': {
                'method': 'POST',
                'description': '图像embedding（原有接口）',
                'body': {'images': '[...] - 图片URL或base64列表'}
            },
            '/embed_text': {
                'method': 'POST',
                'description': '文本embedding（原有接口）',
                'body': {'texts': '[...] 或 text: "..."'}
            },
            '/health': {
                'method': 'GET',
                'description': '健康检查'
            },
            '/status': {
                'method': 'GET',
                'description': '服务状态（并发情况）'
            }
        },
        'rate_limit': {
            'max_concurrent': MAX_CONCURRENT,
            'max_waiting': MAX_WAITING,
            'wait_timeout': WAIT_TIMEOUT
        },
        'image_download': {
            'retries': IMAGE_DOWNLOAD_RETRIES,
            'timeout': IMAGE_DOWNLOAD_TIMEOUT,
            'retry_delay': IMAGE_DOWNLOAD_RETRY_DELAY
        }
    })


if __name__ == '__main__':
    print(f"=" * 60)
    print(f"Embedding Service v2.0.0")
    print(f"=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Rate limit: max_concurrent={MAX_CONCURRENT}, max_waiting={MAX_WAITING}, timeout={WAIT_TIMEOUT}s")
    print(f"Image download: retries={IMAGE_DOWNLOAD_RETRIES}, timeout={IMAGE_DOWNLOAD_TIMEOUT}s")
    print(f"=" * 60)
    app.run(host=HOST, port=PORT, threaded=True)
