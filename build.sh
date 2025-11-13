#!/bin/bash
# 构建脚本 - 用于构建CPU和GPU版本的Docker镜像

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building Embedding Service Docker Images"
echo "=========================================="

# 构建CPU版本
echo ""
echo "Building CPU version..."
docker build -f cpu/Dockerfile -t embedding-service:cpu -t embedding-service:cpu-latest .

# 构建GPU版本
echo ""
echo "Building GPU version..."
docker build -f gpu/Dockerfile -t embedding-service:gpu -t embedding-service:gpu-latest .

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Available images:"
docker images | grep embedding-service
echo ""
echo "To run CPU version:"
echo "  docker run -d -p 8080:8080 --name embedding-cpu embedding-service:cpu"
echo ""
echo "To run GPU version:"
echo "  docker run -d --gpus all -p 8081:8080 --name embedding-gpu embedding-service:gpu"
echo ""

