#!/bin/bash
# Run Training using Docker-in-Docker (DinD)
# ===========================================

echo "============================================================"
echo "Multi-Server ML Training Demo - Docker-in-Docker"
echo "============================================================"

# Create necessary directories
mkdir -p data checkpoints logs

# Stop any existing containers
docker ps -a | grep dind-host | awk '{print $1}' | xargs -r docker rm -f

# Build and run the host container with Docker-in-Docker
echo "Building host container with Docker-in-Docker..."
docker build -f Dockerfile.host -t dind-host:latest .

echo ""
echo "Starting training in Docker-in-Docker mode..."
echo "This will create a host container running Docker, which will"
echo "in turn run 4 worker containers inside it."
echo ""

docker run --rm --privileged \
    --name dind-host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$(pwd):/app" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/logs:/app/logs" \
    dind-host:latest

