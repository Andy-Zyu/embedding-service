.PHONY: build-cpu build-gpu build all run-cpu run-gpu stop clean test help

# 默认目标
help:
	@echo "Embedding Service Docker Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build-cpu    - Build CPU version Docker image"
	@echo "  build-gpu    - Build GPU version Docker image"
	@echo "  build        - Build both CPU and GPU versions"
	@echo "  run-cpu      - Run CPU version container"
	@echo "  run-gpu      - Run GPU version container"
	@echo "  stop         - Stop all containers"
	@echo "  clean        - Remove all containers and images"
	@echo "  test         - Test API endpoints"
	@echo "  help         - Show this help message"

# 构建CPU版本
build-cpu:
	docker build -f cpu/Dockerfile -t embedding-service:cpu -t embedding-service:cpu-latest .

# 构建GPU版本
build-gpu:
	docker build -f gpu/Dockerfile -t embedding-service:gpu -t embedding-service:gpu-latest .

# 构建所有版本
build: build-cpu build-gpu

# 运行CPU版本
run-cpu:
	docker run -d --name embedding-service-cpu -p 8080:8080 embedding-service:cpu

# 运行GPU版本
run-gpu:
	docker run -d --name embedding-service-gpu --gpus all -p 8081:8080 embedding-service:gpu

# 停止所有容器
stop:
	docker stop embedding-service-cpu embedding-service-gpu 2>/dev/null || true

# 清理
clean: stop
	docker rm embedding-service-cpu embedding-service-gpu 2>/dev/null || true
	docker rmi embedding-service:cpu embedding-service:gpu embedding-service:cpu-latest embedding-service:gpu-latest 2>/dev/null || true

# 测试API
test:
	@./test_api.sh

