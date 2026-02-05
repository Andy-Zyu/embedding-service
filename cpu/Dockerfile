# CPU版本的生产环境Dockerfile（使用Gunicorn）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 配置apt使用国内镜像源（阿里云）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    (echo "deb http://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
     echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
     echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list)

# 配置pip使用国内镜像源（清华大学镜像）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖（CPU版本，不安装CUDA相关的包）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py app_gunicorn.py ./

# 设置环境变量
ENV DEFAULT_MODEL_NAME=google/siglip2-so400m-patch16-naflex
ENV PORT=8080
ENV HOST=0.0.0.0
ENV WORKERS=4
ENV WORKER_CLASS=sync
ENV THREADS=2
ENV TIMEOUT=120
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
# 使用HuggingFace国内镜像
ENV HF_ENDPOINT=https://hf-mirror.com

# 创建缓存目录
RUN mkdir -p /app/.cache/huggingface

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 使用Gunicorn启动（生产环境）
CMD ["gunicorn", "--config", "app_gunicorn.py", "app:app"]
