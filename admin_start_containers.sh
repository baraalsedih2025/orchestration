#!/bin/bash
# ============================================
# Mentor/Admin Script: Start Docker Containers
# ============================================
# This script starts all 4 Docker containers
# Run this with docker permissions (mentor/admin only)

set -e

echo "============================================================"
echo "Starting 4 Docker Containers"
echo "============================================================"

# Check if containers are already running
if docker ps | grep -q "ml-worker"; then
    echo "⚠️  Containers are already running!"
    docker ps | grep "ml-worker"
    echo ""
    read -p "Do you want to restart them? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose.manual.yml down
    else
        echo "Containers already running. Exiting."
        exit 0
    fi
fi

# Start all 4 containers
echo "Starting containers..."
docker-compose -f docker-compose.manual.yml up -d

# Wait a moment for containers to initialize
sleep 3

# Check status
echo ""
echo "Container Status:"
docker-compose -f docker-compose.manual.yml ps

echo ""
echo "✅ All 4 containers started successfully!"
echo ""
echo "Containers are running in the background (sleep infinity)."
echo "User can now run training from their account."
echo ""
echo "To view logs:"
echo "  docker logs -f ml-worker-0"
echo ""
echo "To stop containers:"
echo "  ./admin_stop_containers.sh"
echo "============================================================"

