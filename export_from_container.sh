#!/bin/bash
# 从现有容器导出Docker镜像的脚本

set -e

CONTAINER_NAME="${1:-embedding-svc}"
IMAGE_NAME_CPU="${2:-embedding-service:cpu-from-container}"
IMAGE_NAME_GPU="${3:-embedding-service:gpu-from-container}"

echo "=========================================="
echo "Exporting Docker image from container: $CONTAINER_NAME"
echo "=========================================="

# 检查容器是否存在
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' not found!"
    echo "Available containers:"
    docker ps -a --format '{{.Names}}'
    exit 1
fi

# 检查容器是否运行
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container is running. Stopping it first..."
    docker stop "$CONTAINER_NAME"
fi

# 提交容器为镜像
echo ""
echo "Committing container to image..."
docker commit "$CONTAINER_NAME" "$IMAGE_NAME_CPU"

# 检查是否有GPU支持
if docker exec "$CONTAINER_NAME" python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU support detected. Creating GPU version tag..."
    docker tag "$IMAGE_NAME_CPU" "$IMAGE_NAME_GPU"
    echo "Created GPU version: $IMAGE_NAME_GPU"
fi

echo ""
echo "=========================================="
echo "Export completed!"
echo "=========================================="
echo ""
echo "Created images:"
docker images | grep -E "(${IMAGE_NAME_CPU}|${IMAGE_NAME_GPU})" | head -5
echo ""
echo "To run the exported image:"
echo "  docker run -d -p 8080:8080 --name embedding-test $IMAGE_NAME_CPU"
echo ""
echo "To save image to file:"
echo "  docker save $IMAGE_NAME_CPU | gzip > embedding-service.tar.gz"
echo ""

