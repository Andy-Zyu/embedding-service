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
ENABLE_TEST=false
TEST_TYPE=""
TEST_CONCURRENCY=0
TEST_IMAGE_PATH="test/images/test.png"

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

# 构建镜像
build_image() {
    echo ""
    print_info "开始构建镜像..."
    
    if [ "$VERSION" == "cpu" ]; then
        docker build -f cpu/Dockerfile -t embedding-service:cpu .
        print_success "CPU镜像构建完成"
    else
        docker build --platform linux/amd64 -f gpu/Dockerfile -t embedding-service:gpu .
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
            port=$((8079 + i))
            cat >> "$compose_file" <<EOF
  embedding-service-cpu-${i}:
    image: embedding-service:cpu
    container_name: embedding-service-cpu-${i}
    ports:
      - "${port}:8080"
    environment:
      - MODEL_NAME=google/siglip2-so400m-patch16-naflex
      - PORT=8080
      - HOST=0.0.0.0
      - WORKERS=4
      - THREADS=2
    volumes:
      - huggingface_cache:/app/.cache/huggingface
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
            port=$((8079 + i))
            gpu_id=${GPU_IDS[$((i-1))]}
            workers=$((GPU_MEMORY / 2))
            if [ $workers -lt 1 ]; then
                workers=1
            elif [ $workers -gt 4 ]; then
                workers=4
            fi
            
            cat >> "$compose_file" <<EOF
  embedding-service-gpu-${i}:
    image: embedding-service:gpu
    container_name: embedding-service-gpu-${i}
    ports:
      - "${port}:8080"
    environment:
      - MODEL_NAME=google/siglip2-so400m-patch16-naflex
      - PORT=8080
      - HOST=0.0.0.0
      - WORKERS=${workers}
      - THREADS=4
      - CUDA_VISIBLE_DEVICES=${gpu_id}
    volumes:
      - huggingface_cache:/app/.cache/huggingface
    restart: unless-stopped
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
      - "80:80"
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
    
    cat >> "$compose_file" <<EOF
volumes:
  huggingface_cache:
    driver: local

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
    sleep 10
    
    # 检查服务状态
    local healthy=0
    local max_attempts=30
    for i in $(seq 1 $max_attempts); do
        if [ $INSTANCE_COUNT -gt 1 ]; then
            # 多实例通过Nginx检查
            if curl -s http://localhost/health > /dev/null 2>&1; then
                healthy=1
                break
            fi
        else
            # 单实例直接检查
            if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                healthy=1
                break
            fi
        fi
        if [ $i -lt $max_attempts ]; then
            echo -n "."
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
        print_warning "服务可能还在启动中，请稍后检查"
        print_info "可以使用以下命令查看状态:"
        echo "  docker compose -f docker-compose.deploy.yml ps"
        echo "  docker compose -f docker-compose.deploy.yml logs"
    fi
}

# 运行并发测试
run_benchmark() {
    local url="http://localhost"
    if [ $INSTANCE_COUNT -eq 1 ]; then
        url="http://localhost:8080"
    fi
    
    local concurrency=$1
    local test_type=$2
    
    # 所有输出到stderr，避免污染返回值
    print_info "开始${test_type}并发测试: 并发数=$concurrency" >&2
    
    if [ "$test_type" == "图片" ]; then
        if [ ! -f "$TEST_IMAGE_PATH" ]; then
            print_error "测试图片不存在: $TEST_IMAGE_PATH" >&2
            return 1
        fi
        
        # 转换图片为base64
        print_info "正在转换图片为base64编码..." >&2
        local image_base64=$(base64 -i "$TEST_IMAGE_PATH" 2>/dev/null || base64 "$TEST_IMAGE_PATH" 2>/dev/null)
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
        for i in $(seq 1 $concurrency); do
            (
                local response=$(curl -s -X POST "$url/embed" \
                    -H "Content-Type: application/json" \
                    -d "{\"images\": [\"${image_data}\"]}" \
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
    if [ $INSTANCE_COUNT -gt 1 ]; then
        echo "  通过Nginx: http://localhost"
    else
        echo "  直接访问: http://localhost:8080"
    fi
    echo ""
    echo "管理命令:"
    echo "  查看状态: docker compose -f docker-compose.deploy.yml ps"
    echo "  查看日志: docker compose -f docker-compose.deploy.yml logs -f"
    echo "  停止服务: docker compose -f docker-compose.deploy.yml down"
}

# 运行主函数
main

