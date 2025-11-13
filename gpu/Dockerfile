# GPU版本的Embedding服务Dockerfile
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 设置环境变量
ENV MODEL_NAME=google/siglip2-so400m-patch16-naflex
ENV PORT=8080
ENV HOST=0.0.0.0
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0

# 创建缓存目录
RUN mkdir -p /app/.cache/huggingface

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["python", "app.py"]

