#!/usr/bin/env python3
"""
Embedding Service API
支持图像和文本的embedding向量生成服务
"""
import io
import base64
import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# 模型配置 - 可以通过环境变量覆盖
MODEL_NAME = os.getenv('MODEL_NAME', 'google/siglip2-so400m-patch16-naflex')
PORT = int(os.getenv('PORT', '8080'))
HOST = os.getenv('HOST', '0.0.0.0')

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

def load_image_any(x):
    """加载图像，支持base64、URL或本地路径"""
    if isinstance(x, str) and x.startswith('data:'):
        b64 = x.split(',', 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    img = load_image(x)  # 支持 http(s) URL 或本地路径
    return img.convert('RGB') if hasattr(img, 'mode') and img.mode != 'RGB' else img

def encode_texts(texts):
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

def encode_images(images):
    """编码图像为向量"""
    model, processor = load_model()
    
    # 加载图像
    pil_images = [load_image_any(img) for img in images]
    inputs = processor(images=pil_images, return_tensors='pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    
    return feats.detach().cpu().float().tolist()

# Flask应用
app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def embed():
    """图像embedding接口"""
    try:
        data = request.get_json(force=True)
        if 'images' not in data:
            return jsonify({'error': 'missing images field'}), 400
        
        images = data['images']
        if not isinstance(images, list):
            return jsonify({'error': 'images must be a list'}), 400
        
        embeddings = encode_images(images)
        return jsonify(embeddings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/embed_text', methods=['POST'])
def embed_text():
    """文本embedding接口"""
    try:
        data = request.get_json(force=True)
        texts = data.get('texts') or data.get('text')
        
        if texts is None:
            return jsonify({'error': 'missing texts or text field'}), 400
        
        embeddings = encode_texts(texts)
        return jsonify(embeddings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/', methods=['GET'])
def index():
    """API信息"""
    return jsonify({
        'service': 'Embedding Service',
        'model': MODEL_NAME,
        'endpoints': {
            '/embed': 'POST - 图像embedding (body: {"images": [...]})',
            '/embed_text': 'POST - 文本embedding (body: {"texts": [...]} or {"text": "..."})',
            '/health': 'GET - 健康检查'
        }
    })

if __name__ == '__main__':
    print(f"Starting Embedding Service with model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    app.run(host=HOST, port=PORT, threaded=True)

