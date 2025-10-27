#!/bin/bash
# Run Training on 4 Docker Containers
# ===================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

echo "============================================================"
echo "Multi-Server ML Training Demo"
echo "============================================================"

# Check if docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    print_status $RED "✗ Docker is not installed or not running"
    print_status $YELLOW "Please install Docker and ensure the daemon is running"
    exit 1
fi

# Check for docker compose (v2) or docker-compose (v1)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    print_status $RED "✗ Docker Compose is not installed"
    exit 1
fi
print_status $BLUE "Using: $DOCKER_COMPOSE_CMD"

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    print_status $RED "✗ Docker daemon is not running"
    print_status $YELLOW ""
    print_status $YELLOW "To start Docker daemon, try:"
    print_status $YELLOW "  sudo systemctl start docker"
    print_status $YELLOW "  or"
    print_status $YELLOW "  sudo service docker start"
    print_status $YELLOW ""
    print_status $YELLOW "Alternatively, you can use local testing instead:"
    print_status $YELLOW "  ./test_single.sh"
    exit 1
fi

# Create necessary directories
print_status $BLUE "Creating directories..."
mkdir -p data checkpoints logs
print_status $GREEN "✓ Directories created"

# Stop any existing containers
print_status $BLUE "Stopping any existing containers..."
$DOCKER_COMPOSE_CMD down 2>/dev/null || true

# Set Docker environment
export DOCKER_HOST=unix:///var/run/docker.sock

# Build Docker images
print_status $BLUE "Building Docker images for 4 workers..."
$DOCKER_COMPOSE_CMD build
print_status $GREEN "✓ Docker images built"

# Start 4 Docker containers
print_status $BLUE "Starting 4 Docker containers..."
print_status $YELLOW "  Worker 0 (Master): ml-worker-0"
print_status $YELLOW "  Worker 1:         ml-worker-1"
print_status $YELLOW "  Worker 2:         ml-worker-2"
print_status $YELLOW "  Worker 3:         ml-worker-3"
print_status $YELLOW "  Total: 4 GPUs (1 per worker)"
echo ""

$DOCKER_COMPOSE_CMD up -d

# Wait for containers to be ready
print_status $BLUE "Waiting for containers to initialize..."
sleep 5

# Check if containers are running
if $DOCKER_COMPOSE_CMD ps | grep -q "Up"; then
    print_status $GREEN "✓ All 4 containers started successfully"
    echo ""
    print_status $BLUE "Container Status:"
    $DOCKER_COMPOSE_CMD ps
    echo ""
    print_status $BLUE "Starting training on all 4 workers..."
    print_status $YELLOW "Press Ctrl+C to stop (logs will continue)"
    echo ""
    print_status $BLUE "============================================================"
    print_status $BLUE "Training Logs:"
    print_status $BLUE "============================================================"
    
    # Show logs from all containers
    $DOCKER_COMPOSE_CMD logs -f
    
else
    print_status $RED "✗ Failed to start containers"
    $DOCKER_COMPOSE_CMD logs
    exit 1
fi
