#!/bin/bash
# Quick Run with docker-compose using pre-built images
# =====================================================

set -e

echo "============================================================"
echo "Starting 4 Workers with Pre-built PyTorch Image"
echo "============================================================"

# Make sure Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Pull pre-built image (if permissions allow)
echo "Pulling PyTorch image..."
docker pull pytorch/pytorch:latest 2>&1 | grep -v "operation not permitted" || echo "Note: Image pull limited by permissions"

# Start containers
echo ""
echo "Starting 4 Docker containers..."
docker-compose up -d

echo ""
echo "Containers starting..."
sleep 10

# Show status
docker-compose ps

echo ""
echo "Following logs from all containers..."
docker-compose logs -f
