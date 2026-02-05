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

import json as _json

import requests
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# 模型配置 - 可以通过环境变量覆盖
DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME')
LEGACY_MODEL_NAME = os.getenv('MODEL_NAME')
if not DEFAULT_MODEL_NAME:
    DEFAULT_MODEL_NAME = LEGACY_MODEL_NAME or 'google/siglip2-so400m-patch16-naflex'

def _parse_models(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]

AVAILABLE_MODELS_RAW = os.getenv('AVAILABLE_MODELS', '')
if AVAILABLE_MODELS_RAW:
    AVAILABLE_MODELS = _parse_models(AVAILABLE_MODELS_RAW)
elif LEGACY_MODEL_NAME:
    AVAILABLE_MODELS = _parse_models(LEGACY_MODEL_NAME)
else:
    AVAILABLE_MODELS = [DEFAULT_MODEL_NAME]

if DEFAULT_MODEL_NAME not in AVAILABLE_MODELS:
    AVAILABLE_MODELS.append(DEFAULT_MODEL_NAME)

PRELOAD_MODELS = os.getenv('PRELOAD_MODELS', '0').lower() in ('1', 'true', 'yes', 'y')

SENTENCE_TRANSFORMERS_MODELS_RAW = os.getenv('SENTENCE_TRANSFORMERS_MODELS', '')
SENTENCE_TRANSFORMERS_MODELS = _parse_models(SENTENCE_TRANSFORMERS_MODELS_RAW) if SENTENCE_TRANSFORMERS_MODELS_RAW else []

# Marqo FashionSigLIP 模型列表 (使用 open_clip 后端，API 不同)
MARQO_FASHION_MODELS_RAW = os.getenv('MARQO_FASHION_MODELS', 'Marqo/marqo-fashionSigLIP')
MARQO_FASHION_MODELS = _parse_models(MARQO_FASHION_MODELS_RAW) if MARQO_FASHION_MODELS_RAW else []
PORT = int(os.getenv('PORT', '8080'))
HOST = os.getenv('HOST', '0.0.0.0')

# ============ 输入类型识别/校验配置 ============
# 背景：/v1/embeddings 扩展支持 image，但 OpenAI 官方协议里 embeddings 仅文本。
# 如果调用方忘传 input_type=image（默认 text），图片URL/base64 会被当作“文本字符串”编码，
# 容易产生“向量塌缩/高度相似”的错觉（尤其是URL前缀相似时）。
AUTO_DETECT_INPUT_TYPE = os.getenv('AUTO_DETECT_INPUT_TYPE', '0').lower() in ('1', 'true', 'yes', 'y')
REJECT_MISMATCH_INPUT_TYPE = os.getenv('REJECT_MISMATCH_INPUT_TYPE', '0').lower() in ('1', 'true', 'yes', 'y')

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
_model_cache = {}
_processor_cache = {}
_backend_cache = {}
_model_lock = threading.Lock()


def _is_sentence_transformer(model_name: str) -> bool:
    return model_name in SENTENCE_TRANSFORMERS_MODELS


def _is_marqo_fashion_model(model_name: str) -> bool:
    return model_name in MARQO_FASHION_MODELS


def _find_marqo_local_path(model_name: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """尝试在本地查找 Marqo 模型目录（直接下载目录，非 hub 缓存格式）"""
    candidates = []
    if cache_dir:
        candidates.append(os.path.join(cache_dir, model_name))
        candidates.append(os.path.join(cache_dir, 'hub', model_name))
    hf_home = os.getenv('HF_HOME')
    if hf_home and hf_home != cache_dir:
        candidates.append(os.path.join(hf_home, model_name))
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'open_clip_config.json')):
            return path
    return None


def _ensure_marqo_hub_cache(model_name: str, local_path: str, hub_cache_dir: Optional[str] = None):
    """确保 hub 缓存中包含 open_clip 所需的文件（从本地直接下载目录同步）。
    
    open_clip 使用 huggingface_hub 的缓存格式，如果 hub 缓存中缺少文件（例如之前
    snapshot_download 被中断），这里会从已有的本地目录创建缓存条目。
    """
    if not hub_cache_dir:
        return

    # hub 缓存格式: {hub_cache_dir}/models--{org}--{name}/snapshots/{revision}/
    safe_name = model_name.replace('/', '--')
    model_hub_dir = os.path.join(hub_cache_dir, f'models--{safe_name}')

    # 查找 snapshot 目录
    snapshots_dir = os.path.join(model_hub_dir, 'snapshots')
    if not os.path.isdir(snapshots_dir):
        os.makedirs(snapshots_dir, exist_ok=True)

    # 查找或创建 snapshot 子目录
    snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if snapshot_dirs:
        snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[0])
    else:
        snapshot_path = os.path.join(snapshots_dir, 'local')
        os.makedirs(snapshot_path, exist_ok=True)

    # 需要同步到 hub 缓存的关键文件
    key_files = [
        'open_clip_config.json',
        'open_clip_model.safetensors',
        'open_clip_pytorch_model.bin',
        'config.json',
        'preprocessor_config.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'special_tokens_map.json',
        'spiece.model',
    ]

    synced = 0
    for fname in key_files:
        src = os.path.join(local_path, fname)
        dst = os.path.join(snapshot_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                synced += 1
            except OSError:
                # 如果 symlink 失败（跨文件系统等），尝试复制
                import shutil
                try:
                    shutil.copy2(src, dst)
                    synced += 1
                except Exception:
                    pass

    # 确保 refs/main 指向 snapshot
    refs_dir = os.path.join(model_hub_dir, 'refs')
    os.makedirs(refs_dir, exist_ok=True)
    refs_main = os.path.join(refs_dir, 'main')
    snapshot_name = os.path.basename(snapshot_path)
    if not os.path.exists(refs_main):
        try:
            with open(refs_main, 'w') as f:
                f.write(snapshot_name)
        except Exception:
            pass

    if synced > 0:
        print(f"  Synced {synced} files to hub cache")


def load_model(model_name: str):
    """加载模型和处理器（按需缓存）"""
    if model_name in _model_cache:
        return _model_cache[model_name], _processor_cache.get(model_name), _backend_cache.get(model_name)

    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name], _processor_cache.get(model_name), _backend_cache.get(model_name)

        cache_dir = os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE')
        local_files_only = os.getenv('LOCAL_FILES_ONLY', '0').lower() in ('1', 'true', 'yes', 'y')

        print(f"Loading model: {model_name}")
        if _is_sentence_transformer(model_name):
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError("sentence-transformers is required for this model") from e

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                cache_folder=cache_dir,
                local_files_only=local_files_only
            )
            _model_cache[model_name] = model
            _backend_cache[model_name] = 'sentence-transformers'
            print(f"Model loaded successfully on device: {device}")
            return model, None, 'sentence-transformers'

        # ---- Marqo FashionSigLIP: 使用 open_clip 直接加载 ----
        # AutoModel.from_pretrained 会触发 meta tensor 错误（accelerate 兼容性问题），
        # 因此 Marqo 模型走 open_clip 原生加载路径。
        if _is_marqo_fashion_model(model_name):
            try:
                import open_clip
            except ImportError as e:
                raise RuntimeError(
                    "open_clip_torch is required for Marqo FashionSigLIP model. "
                    "Install with: pip install open_clip_torch ftfy"
                ) from e

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            hub_cache_dir = os.path.join(cache_dir, 'hub') if cache_dir else None

            # 检查本地直接下载目录，如果有则先将文件链接到 hub 缓存
            local_path = _find_marqo_local_path(model_name, cache_dir)
            if local_path:
                print(f"  Found local Marqo model at: {local_path}")
                _ensure_marqo_hub_cache(model_name, local_path, hub_cache_dir)

            print(f"  Loading via open_clip (hf-hub:{model_name})...")
            model, _, preprocess_val = open_clip.create_model_and_transforms(
                f'hf-hub:{model_name}',
                cache_dir=hub_cache_dir,
            )
            tokenizer = open_clip.get_tokenizer(f'hf-hub:{model_name}')
            model = model.to(device).eval()

            _model_cache[model_name] = model
            # open_clip backend: processor 是 (preprocess_val, tokenizer) 元组
            _processor_cache[model_name] = (preprocess_val, tokenizer)
            _backend_cache[model_name] = 'open_clip'
            print(f"Marqo model loaded successfully on device: {device}")
            return model, _processor_cache[model_name], 'open_clip'

        # ---- 标准 transformers 模型 ----
        device_map = 'auto' if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        _model_cache[model_name] = model
        _processor_cache[model_name] = processor
        _backend_cache[model_name] = 'transformers'
        print(f"Model loaded successfully on device: {next(model.parameters()).device}")

    return model, processor, _backend_cache.get(model_name, 'transformers')


def _resolve_model_name(requested: Optional[str]) -> str:
    if not requested:
        return DEFAULT_MODEL_NAME
    if requested in AVAILABLE_MODELS:
        return requested
    return ''


def _preload_models_if_needed():
    if not PRELOAD_MODELS:
        return
    for name in AVAILABLE_MODELS:
        load_model(name)


_preload_models_if_needed()


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


def _looks_like_image_string(x: str) -> bool:
    """粗略判断一个字符串是否“看起来像图片输入”(用于 /v1/embeddings 的防误用)"""
    if not isinstance(x, str):
        return False
    s = x.strip()
    if not s:
        return False
    if s.startswith('data:image/'):
        return True
    if s.startswith(('http://', 'https://')):
        try:
            path = urlparse(s).path.lower()
        except Exception:
            path = s.lower()
        for ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff'):
            if path.endswith(ext):
                return True
        return False
    # 本地路径/文件名：仅根据扩展名判断（可能误判，但概率较低）
    lower = s.lower()
    return any(lower.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff'))


def _maybe_autodetect_input_type(inputs: List[str]) -> Optional[str]:
    """如果 inputs 全部看起来像图片输入，则返回 'image'；否则返回 None（表示保持默认text）"""
    if not inputs:
        return None
    if all(isinstance(x, str) and _looks_like_image_string(x) for x in inputs):
        return 'image'
    return None


def encode_texts(texts: Union[str, List[str]], model_name: str) -> List[List[float]]:
    """编码文本为向量"""
    if isinstance(texts, str):
        texts = [texts]

    model, processor, backend = load_model(model_name)
    if backend == 'sentence-transformers':
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    # ---- open_clip 后端（Marqo FashionSigLIP）----
    if backend == 'open_clip':
        preprocess_val, tokenizer = processor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text_tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            feats = model.encode_text(text_tokens, normalize=True)
        return feats.detach().cpu().float().tolist()

    # ---- 标准 transformers 后端 ----
    inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def _mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = torch.sum(masked, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            feats = model.get_text_features(**inputs)
            if not torch.is_tensor(feats):
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = _mean_pooling(feats.last_hidden_state, inputs["attention_mask"])
                else:
                    raise ValueError("Unexpected output type from get_text_features")
        else:
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

    return feats.detach().cpu().float().tolist()


def encode_images(images: List[Union[str, bytes]], model_name: str) -> List[List[float]]:
    """编码图像为向量"""
    model, processor, backend = load_model(model_name)
    if backend == 'sentence-transformers':
        raise ValueError(f"Model '{model_name}' does not support image embeddings")

    # 加载图像（带重试机制）
    pil_images = []
    for idx, img in enumerate(images):
        try:
            pil_images.append(load_image_any(img))
        except Exception as e:
            raise ValueError(f"Failed to load image at index {idx}: {str(e)}")

    # ---- open_clip 后端（Marqo FashionSigLIP）----
    if backend == 'open_clip':
        preprocess_val, tokenizer = processor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        images_tensor = torch.stack([preprocess_val(img) for img in pil_images]).to(device)
        with torch.no_grad():
            feats = model.encode_image(images_tensor, normalize=True)
        return feats.detach().cpu().float().tolist()

    # ---- 标准 transformers 后端 ----
    inputs = processor(images=pil_images, return_tensors='pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def _mean_pooling_image(last_hidden_state):
        return last_hidden_state.mean(dim=1)

    with torch.no_grad():
        if not hasattr(model, "get_image_features"):
            raise ValueError(f"Model '{model_name}' does not support image embeddings")
        feats = model.get_image_features(**inputs)
        if not torch.is_tensor(feats):
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = _mean_pooling_image(feats.last_hidden_state)
            else:
                raise ValueError("Unexpected output type from get_image_features")
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
        "model": "可选，指定模型（需在可用列表中）",
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
        
        # input_type 基础校验
        if input_type not in ('text', 'image'):
            return jsonify({
                'error': {
                    'message': "input_type must be 'text' or 'image'",
                    'type': 'invalid_request_error',
                    'code': 'invalid_input_type'
                }
            }), 400

        # 可选：自动识别 image（避免忘传 input_type=image 造成误用）
        if 'input_type' not in data and AUTO_DETECT_INPUT_TYPE:
            detected = _maybe_autodetect_input_type(inputs)
            if detected is not None:
                input_type = detected

        # 可选：强校验，拒绝“看起来像图片”的文本请求（避免把图片URL/base64当文本造成疑似塌缩）
        if REJECT_MISMATCH_INPUT_TYPE and input_type == 'text':
            if any(isinstance(x, str) and _looks_like_image_string(x) for x in inputs):
                return jsonify({
                    'error': {
                        'message': "input looks like image (url/base64/path). If you want image embeddings, set input_type='image'.",
                        'type': 'invalid_request_error',
                        'code': 'input_type_mismatch'
                    }
                }), 400

        # 模型选择
        requested_model = data.get('model')
        model_name = _resolve_model_name(requested_model)
        if not model_name:
            return jsonify({
                'error': {
                    'message': f"Unknown model '{requested_model}'. Available models: {', '.join(AVAILABLE_MODELS)}",
                    'type': 'invalid_request_error',
                    'code': 'invalid_model'
                }
            }), 400

        # 根据类型编码
        if input_type == 'image':
            embeddings = encode_images(inputs, model_name)
        else:
            embeddings = encode_texts(inputs, model_name)
        
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
            'model': model_name,
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
        
        embeddings = encode_images(images, DEFAULT_MODEL_NAME)
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
        
        embeddings = encode_texts(texts, DEFAULT_MODEL_NAME)
        return jsonify(embeddings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 管理接口 ============
@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model': DEFAULT_MODEL_NAME,
        'available_models': AVAILABLE_MODELS,
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
        'model': DEFAULT_MODEL_NAME,
        'available_models': AVAILABLE_MODELS,
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
    print(f"Default model: {DEFAULT_MODEL_NAME}")
    print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Rate limit: max_concurrent={MAX_CONCURRENT}, max_waiting={MAX_WAITING}, timeout={WAIT_TIMEOUT}s")
    print(f"Image download: retries={IMAGE_DOWNLOAD_RETRIES}, timeout={IMAGE_DOWNLOAD_TIMEOUT}s")
    print(f"=" * 60)
    app.run(host=HOST, port=PORT, threaded=True)
