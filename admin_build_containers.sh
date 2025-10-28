#!/bin/bash
# ============================================
# Mentor/Admin Script: Build Docker Containers
# ============================================
# This script builds the Docker images for all 4 workers
# Run this with docker permissions (mentor/admin only)

set -e

echo "============================================================"
echo "Building Docker Images for 4 Workers"
echo "============================================================"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose not found"
    exit 1
fi

# Build Docker images for all workers
echo "Building images..."
docker-compose -f docker-compose.manual.yml build

echo ""
echo "✅ Docker images built successfully!"
echo ""
echo "Containers ready to be started with:"
echo "  ./admin_start_containers.sh"
echo "============================================================"

