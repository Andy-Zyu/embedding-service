#!/bin/bash
# Embedding Service 一键部署和测试脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
VERSION=""
GPU_COUNT=0
GPU_IDS=()
GPU_MEMORY=0
INSTANCE_COUNT=0
SERVICE_PORT=18730
ENABLE_TEST=false
TEST_TYPE=""
TEST_CONCURRENCY=0
TEST_IMAGE_PATH="test/images/test.png"
MODEL_DOWNLOAD_MODE=""  # "host" 或 "container"
HF_CACHE_DIR=""  # 宿主机HuggingFace缓存目录
DEFAULT_MODEL_NAME="google/siglip2-so400m-patch16-naflex"
AVAILABLE_MODELS="google/siglip2-so400m-patch16-naflex,infgrad/stella-mrl-large-zh-v3.5-1792d,Marqo/marqo-fashionSigLIP"
PRELOAD_MODELS="0"
SENTENCE_TRANSFORMERS_MODELS="infgrad/stella-mrl-large-zh-v3.5-1792d"
MARQO_FASHION_MODELS="Marqo/marqo-fashionSigLIP"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker和Docker Compose
check_dependencies() {
    print_info "检查依赖..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    print_success "依赖检查通过"
}

# 检查GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            print_success "检测到 $GPU_COUNT 个GPU"
            nvidia-smi --list-gpus
            return 0
        fi
    fi
    return 1
}

# 选择版本
select_version() {
    echo ""
    print_info "请选择部署版本:"
    echo "  1) CPU版本"
    echo "  2) GPU版本"
    read -p "请输入选择 (1/2): " choice
    
    case $choice in
        1)
            VERSION="cpu"
            print_success "已选择CPU版本"
            ;;
        2)
            if ! check_gpu; then
                print_error "未检测到GPU，无法使用GPU版本"
                exit 1
            fi
            VERSION="gpu"
            print_success "已选择GPU版本"
            ;;
        *)
            print_error "无效选择"
            exit 1
            ;;
    esac
}

# 配置GPU
configure_gpu() {
    if [ "$VERSION" != "gpu" ]; then
        return
    fi
    
    echo ""
    print_info "GPU配置"
    
    # 询问GPU数量
    read -p "需要部署多少个GPU实例 (1-$GPU_COUNT): " gpu_instances
    if [ -z "$gpu_instances" ] || [ "$gpu_instances" -lt 1 ] || [ "$gpu_instances" -gt "$GPU_COUNT" ]; then
        print_error "无效的GPU实例数"
        exit 1
    fi
    
    INSTANCE_COUNT=$gpu_instances
    
    # 如果只有一个GPU，默认使用
    if [ "$GPU_COUNT" -eq 1 ]; then
        GPU_IDS=(0)
        print_info "检测到1个GPU，默认使用GPU 0"
    else
        # 询问使用哪些GPU
        echo ""
        print_info "请选择要使用的GPU序号 (0-$((GPU_COUNT-1))):"
        for i in $(seq 1 $gpu_instances); do
            read -p "GPU实例 $i 使用GPU序号: " gpu_id
            if [ -z "$gpu_id" ] || [ "$gpu_id" -lt 0 ] || [ "$gpu_id" -ge "$GPU_COUNT" ]; then
                print_error "无效的GPU序号"
                exit 1
            fi
            GPU_IDS+=($gpu_id)
        done
    fi
    
    # 询问GPU显存分配
    echo ""
    print_info "GPU显存配置（用于动态调整并发和实例数）"
    read -p "每个GPU实例预计分配的显存 (GB, 例如: 8): " gpu_mem
    if [ -z "$gpu_mem" ] || [ "$gpu_mem" -le 0 ]; then
        print_error "无效的显存值"
        exit 1
    fi
    GPU_MEMORY=$gpu_mem
    
    # 根据显存动态调整配置
    # 假设每个worker需要约2GB显存
    workers_per_instance=$((gpu_mem / 2))
    if [ $workers_per_instance -lt 1 ]; then
        workers_per_instance=1
    elif [ $workers_per_instance -gt 4 ]; then
        workers_per_instance=4
    fi
    
    print_info "根据显存配置，每个实例将使用 $workers_per_instance workers"
    
    # 询问模型下载方式
    echo ""
    print_info "模型下载方式配置"
    echo "  1) 在宿主机先下载模型并挂载（推荐，启动更快）"
    echo "  2) 让Docker容器在运行时下载模型（首次启动较慢）"
    read -p "请选择模型下载方式 (1/2, 默认: 1): " download_choice
    
    if [ -z "$download_choice" ]; then
        download_choice=1
    fi
    
    case $download_choice in
        1)
            MODEL_DOWNLOAD_MODE="host"
            print_success "已选择：宿主机下载模型并挂载"
            
            # 询问缓存目录
            echo ""
            read -p "请输入HuggingFace模型缓存目录路径 (默认: ./hf_cache): " cache_dir
            if [ -z "$cache_dir" ]; then
                HF_CACHE_DIR="./hf_cache"
            else
                HF_CACHE_DIR="$cache_dir"
            fi
            
            # 转换为绝对路径
            if [ ! -d "$HF_CACHE_DIR" ]; then
                mkdir -p "$HF_CACHE_DIR"
            fi
            HF_CACHE_DIR=$(cd "$HF_CACHE_DIR" && pwd)
            
            print_info "模型缓存目录: $HF_CACHE_DIR"
            ;;
        2)
            MODEL_DOWNLOAD_MODE="container"
            print_success "已选择：容器运行时下载模型"
            print_warning "首次启动可能需要2-5分钟下载模型，请耐心等待"
            ;;
        *)
            print_error "无效选择，默认使用宿主机下载方式"
            MODEL_DOWNLOAD_MODE="host"
            HF_CACHE_DIR="./hf_cache"
            if [ ! -d "$HF_CACHE_DIR" ]; then
                mkdir -p "$HF_CACHE_DIR"
            fi
            HF_CACHE_DIR=$(cd "$HF_CACHE_DIR" && pwd)
            ;;
    esac
}

# 配置CPU
configure_cpu() {
    if [ "$VERSION" != "cpu" ]; then
        return
    fi
    
    echo ""
    print_info "CPU配置"
    read -p "需要部署多少个CPU实例 (默认: 1): " cpu_instances
    if [ -z "$cpu_instances" ]; then
        cpu_instances=1
    fi
    INSTANCE_COUNT=$cpu_instances
}

# 配置服务端口
configure_port() {
    echo ""
    print_info "端口配置"
    
    if [ $INSTANCE_COUNT -gt 1 ]; then
        read -p "请输入Nginx负载均衡器端口 (默认: 18730): " port
        if [ -z "$port" ]; then
            SERVICE_PORT=18730
        else
            SERVICE_PORT=$port
        fi
        print_info "Nginx将监听端口: $SERVICE_PORT (暴露到宿主机)"
        print_info "后端 $INSTANCE_COUNT 个实例将在Docker内部网络通信（不暴露端口）"
    else
        read -p "请输入服务端口 (默认: 18730): " port
        if [ -z "$port" ]; then
            SERVICE_PORT=18730
        else
            SERVICE_PORT=$port
        fi
        print_info "服务将监听端口: $SERVICE_PORT"
    fi
}

# 在宿主机下载模型
download_model_on_host() {
    if [ "$MODEL_DOWNLOAD_MODE" != "host" ]; then
        return
    fi
    
    echo ""
    print_info "开始在宿主机下载模型..."
    print_info "默认模型: $DEFAULT_MODEL_NAME"
    print_info "可用模型: $AVAILABLE_MODELS"
    print_info "SentenceTransformers模型: $SENTENCE_TRANSFORMERS_MODELS"
    print_info "缓存目录: $HF_CACHE_DIR"
    
    # 检查Python是否可用
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "未找到Python，无法下载模型"
        print_error "请先安装Python或选择容器内下载方式"
        exit 1
    fi
    
    # 检查transformers库是否安装
    local python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
    fi
    
    print_info "检查transformers库..."
    if ! $python_cmd -c "import transformers" 2>/dev/null; then
        print_warning "transformers库未安装，正在安装..."
        print_info "使用清华大学PyPI镜像源加速安装..."
        # 使用清华大学镜像源安装（与Dockerfile保持一致）
        $python_cmd -m pip install --quiet \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --trusted-host pypi.tuna.tsinghua.edu.cn \
            transformers accelerate torch pillow 2>/dev/null || {
            print_error "无法安装transformers库，请手动安装: pip install transformers accelerate torch pillow"
            print_error "或检查网络连接和镜像源配置"
            exit 1
        }
        print_success "transformers库安装完成"
    fi
    
    # 设置环境变量
    local hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HOME="$HF_CACHE_DIR"
    export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
    export HF_ENDPOINT="$hf_endpoint"
    
    print_info "使用HuggingFace镜像: $hf_endpoint"
    print_info "开始下载模型（这可能需要几分钟）..."
    
    # 下载模型
    HF_ENDPOINT="$hf_endpoint" $python_cmd << PYTHON_SCRIPT
import os
import sys

# 必须在导入transformers之前设置环境变量
hf_endpoint = "$hf_endpoint"
os.environ["HF_ENDPOINT"] = hf_endpoint
os.environ["HF_HOME"] = "$HF_CACHE_DIR"
os.environ["TRANSFORMERS_CACHE"] = "$HF_CACHE_DIR"

# 现在导入transformers库（会使用上面设置的HF_ENDPOINT）
from transformers import AutoModel, AutoProcessor
from huggingface_hub import snapshot_download

model_list_raw = "$AVAILABLE_MODELS"
default_model = "$DEFAULT_MODEL_NAME"
cache_dir = "$HF_CACHE_DIR"
marqo_models_raw = "$MARQO_FASHION_MODELS"
marqo_models = [m.strip() for m in marqo_models_raw.split(",") if m.strip()]

model_names = [m.strip() for m in model_list_raw.split(",") if m.strip()]
if not model_names:
    model_names = [default_model]
elif default_model not in model_names:
    model_names.append(default_model)

print(f"Downloading models: {', '.join(model_names)}")
print(f"Cache directory: {cache_dir}")
print(f"HuggingFace endpoint: {hf_endpoint}")
print("This may take several minutes, please wait...", flush=True)

try:
    for model_name in model_names:
        print("=" * 60)
        print(f"Downloading model: {model_name}", flush=True)
        print("=" * 60)

        if model_name in marqo_models:
            print("Downloading model snapshot (Marqo)...", flush=True)
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            print("Model snapshot downloaded successfully!", flush=True)
        else:
            # 下载模型
            print("Downloading model weights...", flush=True)
            AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                local_files_only=False
            )
            print("Model weights downloaded successfully!", flush=True)

            # 下载processor
            print("Downloading processor...", flush=True)
            AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                local_files_only=False
            )
            print("Processor downloaded successfully!", flush=True)
    
    print("=" * 60)
    print("Model download completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"ERROR: Failed to download model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        print_success "模型下载完成！"
        
        # 显示缓存目录大小
        if command -v du &> /dev/null; then
            local cache_size=$(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1)
            print_info "缓存目录大小: $cache_size"
        fi
    else
        print_error "模型下载失败，请检查网络连接或手动下载"
        exit 1
    fi
}

# 构建镜像
build_image() {
    echo ""
    print_info "开始构建镜像..."
    
    if [ "$VERSION" == "cpu" ]; then
        docker build -f cpu/Dockerfile -t embedding-service:cpu .
        print_success "CPU镜像构建完成"
    else
        # GPU版本构建
        if [ "$MODEL_DOWNLOAD_MODE" == "host" ]; then
            # 如果选择宿主机下载，使用build arg跳过Dockerfile中的模型下载
            print_info "检测到宿主机下载模式，跳过Dockerfile中的模型下载步骤..."
            docker build --platform linux/amd64 \
                --build-arg SKIP_MODEL_DOWNLOAD=true \
                -f gpu/Dockerfile \
                -t embedding-service:gpu .
        else
            # 容器内下载模式，正常构建（会在Dockerfile中下载模型）
            docker build --platform linux/amd64 -f gpu/Dockerfile -t embedding-service:gpu .
        fi
        print_success "GPU镜像构建完成"
    fi
}

# 生成docker-compose配置
generate_compose() {
    local compose_file="docker-compose.deploy.yml"
    
    cat > "$compose_file" <<EOF
version: '3.8'

services:
EOF

    if [ "$VERSION" == "cpu" ]; then
        # CPU版本配置
        for i in $(seq 1 $INSTANCE_COUNT); do
            # 单实例时暴露端口到宿主机，多实例时只在内部网络通信
            if [ $INSTANCE_COUNT -eq 1 ]; then
                cat >> "$compose_file" <<EOF
  embedding-service-cpu-${i}:
    image: embedding-service:cpu
    container_name: embedding-service-cpu-${i}
    ports:
      - "${SERVICE_PORT}:8080"
    environment:
EOF
            else
                cat >> "$compose_file" <<EOF
  embedding-service-cpu-${i}:
    image: embedding-service:cpu
    container_name: embedding-service-cpu-${i}
    environment:
EOF
            fi
            
            cat >> "$compose_file" <<EOF
      - DEFAULT_MODEL_NAME=${DEFAULT_MODEL_NAME}
      - AVAILABLE_MODELS=${AVAILABLE_MODELS}
      - PRELOAD_MODELS=${PRELOAD_MODELS}
      - SENTENCE_TRANSFORMERS_MODELS=${SENTENCE_TRANSFORMERS_MODELS}
      - MARQO_FASHION_MODELS=${MARQO_FASHION_MODELS}
      - PORT=8080
      - HOST=0.0.0.0
      - WORKERS=4
      - THREADS=2
    volumes:
EOF
            # 根据下载方式选择volume配置
            if [ "$MODEL_DOWNLOAD_MODE" == "host" ] && [ ! -z "$HF_CACHE_DIR" ]; then
                cat >> "$compose_file" <<EOF
      - ${HF_CACHE_DIR}:/app/.cache/huggingface
EOF
            else
                cat >> "$compose_file" <<EOF
      - huggingface_cache:/app/.cache/huggingface
EOF
            fi

            cat >> "$compose_file" <<EOF
    restart: unless-stopped
    networks:
      - embedding-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

EOF
        done
    else
        # GPU版本配置
        for i in $(seq 1 $INSTANCE_COUNT); do
            gpu_id=${GPU_IDS[$((i-1))]}
            workers=$((GPU_MEMORY / 2))
            if [ $workers -lt 1 ]; then
                workers=1
            elif [ $workers -gt 4 ]; then
                workers=4
            fi
            
            # 单实例时暴露端口到宿主机，多实例时只在内部网络通信
            if [ $INSTANCE_COUNT -eq 1 ]; then
                cat >> "$compose_file" <<EOF
  embedding-service-gpu-${i}:
    image: embedding-service:gpu
    container_name: embedding-service-gpu-${i}
    ports:
      - "${SERVICE_PORT}:8080"
    environment:
EOF
            else
                cat >> "$compose_file" <<EOF
  embedding-service-gpu-${i}:
    image: embedding-service:gpu
    container_name: embedding-service-gpu-${i}
    environment:
EOF
            fi
            
            cat >> "$compose_file" <<EOF
      - DEFAULT_MODEL_NAME=${DEFAULT_MODEL_NAME}
      - AVAILABLE_MODELS=${AVAILABLE_MODELS}
      - PRELOAD_MODELS=${PRELOAD_MODELS}
      - SENTENCE_TRANSFORMERS_MODELS=${SENTENCE_TRANSFORMERS_MODELS}
      - MARQO_FASHION_MODELS=${MARQO_FASHION_MODELS}
      - PORT=8080
      - HOST=0.0.0.0
      - WORKERS=${workers}
      - THREADS=4
      - CUDA_VISIBLE_DEVICES=0
    volumes:
EOF
            # 根据下载方式选择volume配置
            if [ "$MODEL_DOWNLOAD_MODE" == "host" ] && [ ! -z "$HF_CACHE_DIR" ]; then
                cat >> "$compose_file" <<EOF
      - ${HF_CACHE_DIR}:/app/.cache/huggingface
EOF
            else
                cat >> "$compose_file" <<EOF
      - huggingface_cache:/app/.cache/huggingface
EOF
            fi
            
            cat >> "$compose_file" <<EOF
    restart: unless-stopped
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['${gpu_id}']
              capabilities: [gpu]
    networks:
      - embedding-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

EOF
        done
    fi
    
    # 添加Nginx负载均衡（如果多个实例）
    if [ $INSTANCE_COUNT -gt 1 ]; then
        cat >> "$compose_file" <<EOF
  nginx-lb:
    image: nginx:alpine
    container_name: embedding-nginx-lb
    ports:
      - "${SERVICE_PORT}:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
EOF
        if [ "$VERSION" == "cpu" ]; then
            for i in $(seq 1 $INSTANCE_COUNT); do
                echo "      - embedding-service-cpu-${i}" >> "$compose_file"
            done
        else
            for i in $(seq 1 $INSTANCE_COUNT); do
                echo "      - embedding-service-gpu-${i}" >> "$compose_file"
            done
        fi
        cat >> "$compose_file" <<EOF
    restart: unless-stopped
    networks:
      - embedding-network

EOF
    fi
    
    # 只有在使用Docker volume时才定义volumes
    if [ "$MODEL_DOWNLOAD_MODE" != "host" ] || [ -z "$HF_CACHE_DIR" ]; then
        cat >> "$compose_file" <<EOF
volumes:
  huggingface_cache:
    driver: local

EOF
    fi
    
    cat >> "$compose_file" <<EOF
networks:
  embedding-network:
    driver: bridge
EOF

    print_success "Docker Compose配置已生成: $compose_file"
}

# 更新Nginx配置
update_nginx_config() {
    if [ $INSTANCE_COUNT -le 1 ]; then
        return
    fi
    
    local backend_name=""
    if [ "$VERSION" == "cpu" ]; then
        backend_name="cpu_backend"
    else
        backend_name="gpu_backend"
    fi
    
    cat > nginx.conf <<EOF
events {
    worker_connections 1024;
}

http {
    resolver 127.0.0.11 valid=30s;
    
    upstream ${backend_name} {
        least_conn;
EOF
    
    if [ "$VERSION" == "cpu" ]; then
        for i in $(seq 1 $INSTANCE_COUNT); do
            echo "        server embedding-service-cpu-${i}:8080 max_fails=3 fail_timeout=30s;" >> nginx.conf
        done
    else
        for i in $(seq 1 $INSTANCE_COUNT); do
            echo "        server embedding-service-gpu-${i}:8080 max_fails=3 fail_timeout=30s;" >> nginx.conf
        done
    fi
    
    cat >> nginx.conf <<EOF
        keepalive 32;
    }

    server {
        listen 80;
        server_name api;

        location / {
            proxy_pass http://${backend_name};
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_connect_timeout 120s;
            proxy_send_timeout 120s;
            proxy_read_timeout 120s;
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }
    }
}
EOF
    
    print_success "Nginx配置已更新"
}

# 停止所有旧服务
stop_old_services() {
    echo ""
    print_info "停止并清理旧服务（实现自动更新部署）..."
    
    # 停止当前部署的服务
    if [ -f "docker-compose.deploy.yml" ]; then
        print_info "停止 docker-compose.deploy.yml 服务..."
        docker compose -f docker-compose.deploy.yml down 2>/dev/null || true
    fi
    
    # 停止docker-compose.yml启动的服务
    if [ -f "docker-compose.yml" ]; then
        print_info "停止 docker-compose.yml 服务..."
        docker compose -f docker-compose.yml down 2>/dev/null || true
    fi
    
    # 停止docker-compose.scale.yml启动的服务
    if [ -f "docker-compose.scale.yml" ]; then
        print_info "停止 docker-compose.scale.yml 服务..."
        docker compose -f docker-compose.scale.yml down 2>/dev/null || true
    fi
    
    # 停止所有embedding-service相关的容器（包括手动启动的）
    local containers=$(docker ps -a --filter "name=embedding-service" --filter "name=embedding-nginx-lb" --format "{{.Names}}" 2>/dev/null)
    if [ ! -z "$containers" ]; then
        print_info "发现旧容器，正在停止并删除..."
        echo "$containers" | while read container; do
            if [ ! -z "$container" ]; then
                docker stop "$container" 2>/dev/null || true
                docker rm "$container" 2>/dev/null || true
            fi
        done
    fi
    
    # 等待容器完全停止
    sleep 2
    
    # 清理未使用的网络（但保留embedding-network如果存在）
    docker network prune -f > /dev/null 2>&1 || true
    
    print_success "旧服务已清理完成，可以开始新部署"
}

# 启动服务
start_services() {
    echo ""
    print_info "启动新服务..."
    
    # 启动新服务
    docker compose -f docker-compose.deploy.yml up -d
    
    print_info "等待服务启动..."
    
    # GPU服务启动需要更长时间（模型加载）
    if [ "$VERSION" == "gpu" ]; then
        print_info "GPU服务启动需要较长时间（加载模型），请耐心等待..."
        print_info "模型加载可能需要2-5分钟，取决于模型大小和网络速度..."
        sleep 30  # 增加初始等待时间
        # GPU服务需要更长的健康检查时间（最多10分钟）
        local max_attempts=300  # 300次 * 2秒 = 600秒 = 10分钟
    else
        sleep 10
        local max_attempts=60  # 60次 * 2秒 = 120秒
    fi
    
    # 检查服务状态
    local healthy=0
    
    print_info "检查服务健康状态..."
    for i in $(seq 1 $max_attempts); do
        # 首先检查容器是否在运行
        local running=$(docker compose -f docker-compose.deploy.yml ps --status running 2>/dev/null | grep -c "embedding-service")
        
        if [ "$running" -gt 0 ]; then
            # 容器在运行，检查健康端点
            local health_response=$(curl -s --connect-timeout 5 --max-time 10 http://localhost:${SERVICE_PORT}/health 2>&1)
            local curl_exit_code=$?
            
            if [ $curl_exit_code -eq 0 ] && echo "$health_response" | grep -q "ok"; then
                healthy=1
                break
            fi
            
            # 如果是GPU服务，每30秒显示一次模型加载进度
            if [ "$VERSION" == "gpu" ] && [ $((i % 15)) -eq 0 ]; then
                echo ""
                print_info "模型仍在加载中... (已等待 $((i * 2)) 秒)"
                # 显示容器日志的最后几行，帮助了解进度
                local last_log=$(docker compose -f docker-compose.deploy.yml logs --tail=3 2>/dev/null | grep -i "model\|loading\|preload" | tail -1)
                if [ ! -z "$last_log" ]; then
                    echo "  最新日志: $last_log"
                fi
            fi
        else
            # 容器未运行，检查是否有错误
            local exited=$(docker compose -f docker-compose.deploy.yml ps --status exited 2>/dev/null | grep -c "embedding-service")
            if [ "$exited" -gt 0 ]; then
                print_error "检测到容器已退出，请检查日志:"
                docker compose -f docker-compose.deploy.yml logs --tail=20
                break
            fi
        fi
        
        if [ $i -lt $max_attempts ]; then
            if [ $((i % 10)) -eq 0 ]; then
                echo -n " [${i}s]"
            else
                echo -n "."
            fi
            sleep 2
        fi
    done
    echo ""
    
    if [ $healthy -eq 1 ]; then
        print_success "服务启动成功"
        
        # 显示服务状态
        echo ""
        print_info "服务状态:"
        docker compose -f docker-compose.deploy.yml ps
    else
        print_warning "健康检查超时，但服务可能仍在启动中"
        
        # 显示当前容器状态
        echo ""
        print_info "当前容器状态:"
        docker compose -f docker-compose.deploy.yml ps
        
        echo ""
        print_info "查看详细日志的命令:"
        echo "  docker compose -f docker-compose.deploy.yml logs -f"
        
        echo ""
        print_info "如果是GPU服务，模型加载可能需要更长时间"
        print_info "您可以继续等待或手动检查服务状态"
        
        # 询问是否继续等待
        echo ""
        read -p "是否继续等待60秒? (y/n): " continue_wait
        if [ "$continue_wait" == "y" ] || [ "$continue_wait" == "Y" ]; then
            print_info "继续等待..."
            for i in $(seq 1 30); do
                if curl -s --connect-timeout 5 --max-time 10 http://localhost:${SERVICE_PORT}/health > /dev/null 2>&1; then
                    print_success "服务启动成功！"
                    return
                fi
                echo -n "."
                sleep 2
            done
            echo ""
            print_warning "仍未就绪，请手动检查日志"
        fi
    fi
}

# 运行并发测试
run_benchmark() {
    local url="http://localhost:${SERVICE_PORT}"
    
    local concurrency=$1
    local test_type=$2
    
    # 所有输出到stderr，避免污染返回值
    print_info "开始${test_type}并发测试: 并发数=$concurrency" >&2
    
    if [ "$test_type" == "图片" ]; then
        if [ ! -f "$TEST_IMAGE_PATH" ]; then
            print_error "测试图片不存在: $TEST_IMAGE_PATH" >&2
            return 1
        fi
        
        # 转换图片为base64（去除换行符，确保JSON格式正确）
        print_info "正在转换图片为base64编码..." >&2
        local image_base64=$(base64 -w 0 "$TEST_IMAGE_PATH" 2>/dev/null || base64 -i "$TEST_IMAGE_PATH" 2>/dev/null | tr -d '\n' || base64 "$TEST_IMAGE_PATH" 2>/dev/null | tr -d '\n')
        if [ -z "$image_base64" ]; then
            print_error "图片编码失败" >&2
            return 1
        fi
        
        # 根据图片扩展名确定MIME类型
        local image_ext=$(echo "$TEST_IMAGE_PATH" | awk -F. '{print $NF}' | tr '[:upper:]' '[:lower:]')
        local mime_type="image/png"
        case "$image_ext" in
            jpg|jpeg)
                mime_type="image/jpeg"
                ;;
            png)
                mime_type="image/png"
                ;;
            gif)
                mime_type="image/gif"
                ;;
            webp)
                mime_type="image/webp"
                ;;
        esac
        
        local image_data="data:${mime_type};base64,${image_base64}"
        print_info "图片编码完成，大小: $(echo -n "$image_base64" | wc -c | tr -d ' ') 字节" >&2
        
        # 使用curl进行图片embedding测试（benchmark.py不支持图片）
        local success=0
        local failed=0
        local start_time=$(date +%s.%N)
        local temp_file="/tmp/benchmark_times_$$.txt"
        
        > "$temp_file"
        
        print_info "开始发送 $concurrency 个图片embedding请求..." >&2
        # 创建临时JSON文件（避免命令行参数过长）
        local json_temp_file="/tmp/embed_test_$$.json"
        echo "{\"images\": [\"${image_data}\"]}" > "$json_temp_file"
        
        for i in $(seq 1 $concurrency); do
            (
                local response=$(curl -s -X POST "$url/embed" \
                    -H "Content-Type: application/json" \
                    -d @"$json_temp_file" \
                    -w "%{time_total}|%{http_code}" \
                    --max-time 60 \
                    -o /dev/null 2>&1)
                
                local time=$(echo "$response" | cut -d'|' -f1)
                local http_code=$(echo "$response" | cut -d'|' -f2)
                
                if [ "$http_code" == "200" ] && [ ! -z "$time" ] && [ "$time" != "0.000" ]; then
                    echo "$time" >> "$temp_file"
                else
                    echo "ERROR|$http_code" >> "$temp_file"
                fi
            ) &
            
            # 每100个请求显示进度
            if [ $((i % 100)) -eq 0 ]; then
                echo -n "." >&2
            fi
        done
        
        wait
        echo "" >&2
        
        # 清理临时JSON文件
        rm -f "$json_temp_file"
        
        local end_time=$(date +%s.%N)
        local total_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
        local success=$(grep -v "ERROR" "$temp_file" 2>/dev/null | wc -l | tr -d ' ')
        local failed=$(grep "ERROR" "$temp_file" 2>/dev/null | wc -l | tr -d ' ')
        
        # 计算统计信息
        local avg_time=0
        local min_time=0
        local max_time=0
        local p95_time=0
        local p99_time=0
        
        if [ $success -gt 0 ]; then
            local sorted_file="/tmp/benchmark_sorted_$$.txt"
            grep -v "ERROR" "$temp_file" | sort -n > "$sorted_file"
            
            avg_time=$(awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}' "$sorted_file")
            min_time=$(head -1 "$sorted_file")
            max_time=$(tail -1 "$sorted_file")
            
            if [ $success -gt 10 ]; then
                local p95_line=$(awk "BEGIN {line=int(NR*0.95); if(line==0) line=1; print line}" "$sorted_file" | head -1)
                local p99_line=$(awk "BEGIN {line=int(NR*0.99); if(line==0) line=1; print line}" "$sorted_file" | head -1)
                p95_time=$(sed -n "${p95_line}p" "$sorted_file")
                p99_time=$(sed -n "${p99_line}p" "$sorted_file")
            else
                p95_time=$max_time
                p99_time=$max_time
            fi
            
            rm -f "$sorted_file"
        fi
        
        rm -f "$temp_file"
        
        local success_rate=0
        if [ $((success + failed)) -gt 0 ]; then
            success_rate=$(echo "scale=2; $success * 100 / ($success + $failed)" | bc 2>/dev/null || echo "0")
        fi
        
        local rps=0
        local total_time_check=$(echo "$total_time > 0" | bc 2>/dev/null || echo "0")
        if [ "${total_time_check:-0}" -eq 1 ]; then
            rps=$(echo "scale=2; $success / $total_time" | bc 2>/dev/null || echo "0")
        fi
        
        # 返回结果：成功率|成功数|失败数|总耗时|平均时间|最小时间|最大时间|P95|P99|RPS
        echo "$success_rate|$success|$failed|$total_time|$avg_time|$min_time|$max_time|$p95_time|$p99_time|$rps"
    else
        # 文本测试
        local success=0
        local failed=0
        local start_time=$(date +%s.%N)
        local temp_file="/tmp/benchmark_text_times_$$.txt"
        
        > "$temp_file"
        
        print_info "开始发送 $concurrency 个文本embedding请求..." >&2
        for i in $(seq 1 $concurrency); do
            (
                local response=$(curl -s -X POST "$url/embed_text" \
                    -H "Content-Type: application/json" \
                    -d '{"texts": ["Hello world embedding test"]}' \
                    -w "%{time_total}|%{http_code}" \
                    --max-time 30 \
                    -o /dev/null 2>&1)
                
                local time=$(echo "$response" | cut -d'|' -f1)
                local http_code=$(echo "$response" | cut -d'|' -f2)
                
                if [ "$http_code" == "200" ] && [ ! -z "$time" ] && [ "$time" != "0.000" ]; then
                    echo "$time" >> "$temp_file"
                else
                    echo "ERROR|$http_code" >> "$temp_file"
                fi
            ) &
            
            # 每100个请求显示进度
            if [ $((i % 100)) -eq 0 ]; then
                echo -n "." >&2
            fi
        done
        
        wait
        echo "" >&2
        
        local end_time=$(date +%s.%N)
        local total_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
        local success=$(grep -v "ERROR" "$temp_file" 2>/dev/null | wc -l | tr -d ' ')
        local failed=$(grep "ERROR" "$temp_file" 2>/dev/null | wc -l | tr -d ' ')
        
        # 计算统计信息
        local avg_time=0
        local min_time=0
        local max_time=0
        local p95_time=0
        local p99_time=0
        
        if [ $success -gt 0 ]; then
            local sorted_file="/tmp/benchmark_text_sorted_$$.txt"
            grep -v "ERROR" "$temp_file" | sort -n > "$sorted_file"
            
            avg_time=$(awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}' "$sorted_file")
            min_time=$(head -1 "$sorted_file")
            max_time=$(tail -1 "$sorted_file")
            
            if [ $success -gt 10 ]; then
                local p95_line=$(awk "BEGIN {line=int(NR*0.95); if(line==0) line=1; print line}" "$sorted_file" | head -1)
                local p99_line=$(awk "BEGIN {line=int(NR*0.99); if(line==0) line=1; print line}" "$sorted_file" | head -1)
                p95_time=$(sed -n "${p95_line}p" "$sorted_file")
                p99_time=$(sed -n "${p99_line}p" "$sorted_file")
            else
                p95_time=$max_time
                p99_time=$max_time
            fi
            
            rm -f "$sorted_file"
        fi
        
        rm -f "$temp_file"
        
        local success_rate=0
        if [ $((success + failed)) -gt 0 ]; then
            success_rate=$(echo "scale=2; $success * 100 / ($success + $failed)" | bc 2>/dev/null || echo "0")
        fi
        
        local rps=0
        local total_time_check=$(echo "$total_time > 0" | bc 2>/dev/null || echo "0")
        if [ "${total_time_check:-0}" -eq 1 ]; then
            rps=$(echo "scale=2; $success / $total_time" | bc 2>/dev/null || echo "0")
        fi
        
        # 返回结果：成功率|成功数|失败数|总耗时|平均时间|最小时间|最大时间|P95|P99|RPS
        echo "$success_rate|$success|$failed|$total_time|$avg_time|$min_time|$max_time|$p95_time|$p99_time|$rps"
    fi
}

# 执行并发测试流程
execute_test() {
    echo ""
    print_info "并发测试配置"
    read -p "是否启动并发测试? (y/n): " enable_test
    
    if [ "$enable_test" != "y" ] && [ "$enable_test" != "Y" ]; then
        print_info "跳过并发测试"
        return
    fi
    
    echo ""
    echo "请选择测试类型:"
    echo "  1) 文本Embedding"
    echo "  2) 图片Embedding"
    read -p "请输入选择 (1/2): " test_choice
    
    case $test_choice in
        1)
            TEST_TYPE="文本"
            ;;
        2)
            TEST_TYPE="图片"
            if [ ! -f "$TEST_IMAGE_PATH" ]; then
                print_error "测试图片不存在: $TEST_IMAGE_PATH"
                return
            fi
            ;;
        *)
            print_error "无效选择"
            return
            ;;
    esac
    
    read -p "请输入并发数量: " test_concurrency
    if [ -z "$test_concurrency" ] || [ "$test_concurrency" -le 0 ]; then
        print_error "无效的并发数"
        return
    fi
    
    TEST_CONCURRENCY=$test_concurrency
    
    echo ""
    print_info "开始${TEST_TYPE}并发测试..."
    
    local current_concurrency=$TEST_CONCURRENCY
    local best_concurrency=0
    
    while true; do
        local result=$(run_benchmark $current_concurrency "$TEST_TYPE")
        
        # 解析结果（格式：成功率|成功数|失败数|总耗时|平均时间|最小时间|最大时间|P95|P99|RPS）
        local success_rate=$(echo "$result" | cut -d'|' -f1)
        local success=$(echo "$result" | cut -d'|' -f2)
        local failed=$(echo "$result" | cut -d'|' -f3)
        local total_time=$(echo "$result" | cut -d'|' -f4)
        local avg_time=$(echo "$result" | cut -d'|' -f5)
        local min_time=$(echo "$result" | cut -d'|' -f6)
        local max_time=$(echo "$result" | cut -d'|' -f7)
        local p95_time=$(echo "$result" | cut -d'|' -f8)
        local p99_time=$(echo "$result" | cut -d'|' -f9)
        local rps=$(echo "$result" | cut -d'|' -f10)
        
        # 设置默认值，防止空值导致错误
        success_rate=${success_rate:-0}
        success=${success:-0}
        failed=${failed:-0}
        total_time=${total_time:-0}
        avg_time=${avg_time:-0}
        min_time=${min_time:-0}
        max_time=${max_time:-0}
        p95_time=${p95_time:-0}
        p99_time=${p99_time:-0}
        rps=${rps:-0}
        
        echo ""
        echo "=========================================="
        echo "测试结果 (并发数: $current_concurrency, 类型: ${TEST_TYPE})"
        echo "=========================================="
        echo "成功率:        ${success_rate}%"
        echo "成功请求:      $success"
        echo "失败请求:      $failed"
        echo "总耗时:        ${total_time}s"
        echo ""
        
        if [ "${success:-0}" -gt 0 ]; then
            echo "⏱️  响应时间统计 (秒):"
            echo "  平均响应时间:  ${avg_time}"
            echo "  最小响应时间:  ${min_time}"
            echo "  最大响应时间:  ${max_time}"
            local p95_check=$(echo "$p95_time > 0" | bc 2>/dev/null || echo "0")
            if [ "${p95_check:-0}" -eq 1 ]; then
                echo "  P95响应时间:   ${p95_time}"
                echo "  P99响应时间:   ${p99_time}"
            fi
            echo ""
            echo "🚀 性能指标:"
            local rps_check=$(echo "$rps > 0" | bc 2>/dev/null || echo "0")
            if [ "${rps_check:-0}" -eq 1 ] 2>/dev/null; then
                echo "  实际RPS:       ${rps} req/s"
            else
                local calculated_rps=0
                local total_time_check=$(echo "$total_time > 0" | bc 2>/dev/null || echo "0")
                if [ "${total_time_check:-0}" -eq 1 ] 2>/dev/null; then
                    calculated_rps=$(echo "scale=2; $success / $total_time" | bc 2>/dev/null || echo "0")
                fi
                echo "  实际RPS:       ${calculated_rps} req/s"
            fi
        fi
        
        echo "=========================================="
        
        local success_rate_check=$(echo "$success_rate >= 100" | bc 2>/dev/null || echo "0")
        if [ "${success_rate_check:-0}" -eq 1 ] 2>/dev/null; then
            best_concurrency=$current_concurrency
            print_success "测试通过！适合的并发量: $best_concurrency"
            break
        else
            print_warning "失败率过高 (${success_rate}%)"
            read -p "是否降低并发数重新测试? (y/n): " retry
            if [ "$retry" == "y" ] || [ "$retry" == "Y" ]; then
                current_concurrency=$((current_concurrency * 80 / 100))
                if [ "${current_concurrency:-0}" -lt 10 ]; then
                    current_concurrency=10
                fi
                print_info "降低并发数到: $current_concurrency"
            else
                break
            fi
        fi
    done
    
    if [ $best_concurrency -gt 0 ]; then
        echo ""
        print_success "推荐配置:"
        echo "  适合的并发量: $best_concurrency"
        echo "  当前实例数: $INSTANCE_COUNT"
        if [ "$VERSION" == "gpu" ]; then
            echo "  每个实例workers: $((GPU_MEMORY / 2))"
        fi
    fi
}

# 主函数
main() {
    clear
    echo "=========================================="
    echo "  Embedding Service 一键部署脚本"
    echo "=========================================="
    echo ""
    
    check_dependencies
    select_version
    
    if [ "$VERSION" == "gpu" ]; then
        configure_gpu
    else
        configure_cpu
    fi
    
    configure_port
    
    # 如果是GPU版本且选择宿主机下载，先下载模型
    if [ "$VERSION" == "gpu" ]; then
        download_model_on_host
    fi
    
    build_image
    
    if [ $INSTANCE_COUNT -gt 1 ]; then
        update_nginx_config
    fi
    
    generate_compose
    
    # 停止旧服务并启动新服务
    stop_old_services
    start_services
    
    execute_test
    
    echo ""
    print_success "部署完成！"
    echo ""
    echo "服务访问地址:"
    echo "  http://localhost:${SERVICE_PORT}"
    if [ $INSTANCE_COUNT -gt 1 ]; then
        echo "  (通过Nginx负载均衡，后端 ${INSTANCE_COUNT} 个实例)"
    fi
    echo ""
    echo "管理命令:"
    echo "  查看状态: docker compose -f docker-compose.deploy.yml ps"
    echo "  查看日志: docker compose -f docker-compose.deploy.yml logs -f"
    echo "  停止服务: docker compose -f docker-compose.deploy.yml down"
}

# 运行主函数
main

