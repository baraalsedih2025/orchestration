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

if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
    print_status $RED "✗ Docker Compose is not installed"
    exit 1
fi

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
docker-compose down 2>/dev/null || true

# Build Docker images
print_status $BLUE "Building Docker images for 4 workers..."
docker-compose build
print_status $GREEN "✓ Docker images built"

# Start 4 Docker containers
print_status $BLUE "Starting 4 Docker containers..."
print_status $YELLOW "  Worker 0 (Master): ml-worker-0"
print_status $YELLOW "  Worker 1:         ml-worker-1"
print_status $YELLOW "  Worker 2:         ml-worker-2"
print_status $YELLOW "  Worker 3:         ml-worker-3"
print_status $YELLOW "  Total: 4 GPUs (1 per worker)"
echo ""

docker-compose up -d

# Wait for containers to be ready
print_status $BLUE "Waiting for containers to initialize..."
sleep 5

# Check if containers are running
if docker-compose ps | grep -q "Up"; then
    print_status $GREEN "✓ All 4 containers started successfully"
    echo ""
    print_status $BLUE "Container Status:"
    docker-compose ps
    echo ""
    print_status $BLUE "Starting training on all 4 workers..."
    print_status $YELLOW "Press Ctrl+C to stop (logs will continue)"
    echo ""
    print_status $BLUE "============================================================"
    print_status $BLUE "Training Logs:"
    print_status $BLUE "============================================================"
    
    # Show logs from all containers
    docker-compose logs -f
    
else
    print_status $RED "✗ Failed to start containers"
    docker-compose logs
    exit 1
fi
