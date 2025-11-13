.PHONY: build-cpu build-gpu build all up up-scale down stop clean test help

# 默认目标
help:
	@echo "Embedding Service Docker Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build-cpu    - Build CPU version Docker image"
	@echo "  build-gpu    - Build GPU version Docker image"
	@echo "  build        - Build both CPU and GPU versions"
	@echo "  up           - Start single instance with docker compose"
	@echo "  up-scale     - Start multiple instances (3 CPU + 2 GPU)"
	@echo "  down         - Stop all containers"
	@echo "  stop         - Stop all containers (alias for down)"
	@echo "  clean        - Remove all containers and images"
	@echo "  test         - Test API endpoints"
	@echo "  help         - Show this help message"

# 构建CPU版本
build-cpu:
	docker build -f cpu/Dockerfile -t embedding-service:cpu -t embedding-service:cpu-latest .

# 构建GPU版本（指定平台为linux/amd64，用于在Mac上构建Linux镜像）
build-gpu:
	docker build --platform linux/amd64 -f gpu/Dockerfile -t embedding-service:gpu -t embedding-service:gpu-latest .

# 构建所有版本
build: build-cpu build-gpu

# 启动单实例（docker compose）
up:
	docker compose up -d

# 启动多实例横向扩展
up-scale:
	docker compose -f docker-compose.scale.yml up -d \
		--scale embedding-service-cpu=3 \
		--scale embedding-service-gpu=2

# 停止所有容器
down:
	docker compose down
	docker compose -f docker-compose.scale.yml down 2>/dev/null || true

# 停止所有容器（别名）
stop: down

# 清理
clean: down
	docker rmi embedding-service:cpu embedding-service:gpu 2>/dev/null || true

# 测试API
test:
	@./test_api.sh

