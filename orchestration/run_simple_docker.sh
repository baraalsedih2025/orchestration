#!/bin/bash
# Run 4-Worker Distributed Training (No Docker Build - Use Existing Image)
# ========================================================================

set -e

# Colors
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

# Create directories
print_status $BLUE "Creating directories..."
mkdir -p data checkpoints logs
print_status $GREEN "✓ Directories created"

# Use a pre-built PyTorch image
print_status $BLUE "Using pre-built PyTorch image..."
print_status $YELLOW "Note: Docker build requires root privileges in this environment"
print_status $YELLOW "Using pytorch/pytorch:latest image directly"

# Create a simple docker-compose that uses an existing image
cat > docker-compose-simple.yml <<'EOF'
version: '3.8'

services:
  ml-worker-0:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    container_name: ml-worker-0
    working_dir: /app
    environment:
      - MASTER_ADDR=ml-worker-0
      - MASTER_PORT=12355
      - WORLD_SIZE=4
      - RANK=0
      - LOCAL_RANK=0
      - SLURM_PROCID=0
      - SLURM_LOCALID=0
      - SLURM_NTASKS=4
      - SLURM_NODEID=0
      - SLURM_JOB_ID=001
      - SLURM_JOB_NAME=demo
      - SLURM_JOB_NODELIST=ml-worker-0,ml-worker-1,ml-worker-2,ml-worker-3
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./slurm_training_demo.py:/app/slurm_training_demo.py
      - ./config.json:/app/config.json
      - ./requirements.txt:/app/requirements.txt
    networks:
      - ml-network
    command: bash -c "pip install -r requirements.txt && python3 slurm_training_demo.py --config config.json"

  ml-worker-1:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    container_name: ml-worker-1
    depends_on:
      - ml-worker-0
    working_dir: /app
    environment:
      - MASTER_ADDR=ml-worker-0
      - MASTER_PORT=12355
      - WORLD_SIZE=4
      - RANK=1
      - LOCAL_RANK=0
      - SLURM_PROCID=1
      - SLURM_LOCALID=0
      - SLURM_NTASKS=4
      - SLURM_NODEID=1
      - SLURM_JOB_ID=001
      - SLURM_JOB_NAME=demo
      - SLURM_JOB_NODELIST=ml-worker-0,ml-worker-1,ml-worker-2,ml-worker-3
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./slurm_training_demo.py:/app/slurm_training_demo.py
      - ./config.json:/app/config.json
      - ./requirements.txt:/app/requirements.txt
    networks:
      - ml-network
    command: bash -c "pip install -r requirements.txt && python3 slurm_training_demo.py --config config.json"

  ml-worker-2:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    container_name: ml-worker-2
    depends_on:
      - ml-worker-0
    working_dir: /app
    environment:
      - MASTER_ADDR=ml-worker-0
      - MASTER_PORT=12355
      - WORLD_SIZE=4
      - RANK=2
      - LOCAL_RANK=0
      - SLURM_PROCID=2
      - SLURM_LOCALID=0
      - SLURM_NTASKS=4
      - SLURM_NODEID=2
      - SLURM_JOB_ID=001
      - SLURM_JOB_NAME=demo
      - SLURM_JOB_NODELIST=ml-worker-0,ml-worker-1,ml-worker-2,ml-worker-3
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./slurm_training_demo.py:/app/slurm_training_demo.py
      - ./config.json:/app/config.json
      - ./requirements.txt:/app/requirements.txt
    networks:
      - ml-network
    command: bash -c "pip install -r requirements.txt && python3 slurm_training_demo.py --config config.json"

  ml-worker-3:
    image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    container_name: ml-worker-3
    depends_on:
      - ml-worker-0
    working_dir: /app
    environment:
      - MASTER_ADDR=ml-worker-0
      - MASTER_PORT=12355
      - WORLD_SIZE=4
      - RANK=3
      - LOCAL_RANK=0
      - SLURM_PROCID=3
      - SLURM_LOCALID=0
      - SLURM_NTASKS=4
      - SLURM_NODEID=3
      - SLURM_JOB_ID=001
      - SLURM_JOB_NAME=demo
      - SLURM_JOB_NODELIST=ml-worker-0,ml-worker-1,ml-worker-2,ml-worker-3
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./slurm_training_demo.py:/app/slurm_training_demo.py
      - ./config.json:/app/config.json
      - ./requirements.txt:/app/requirements.txt
    networks:
      - ml-network
    command: bash -c "pip install -r requirements.txt && python3 slurm_training_demo.py --config config.json"

networks:
  ml-network:
    driver: bridge
EOF

print_status $GREEN "✓ Created docker-compose-simple.yml"

# Try to start containers
print_status $BLUE "Starting 4 Docker containers..."
docker-compose -f docker-compose-simple.yml up -d

print_status $BLUE "Waiting for containers..."
sleep 10

print_status $BLUE "Container Status:"
docker-compose -f docker-compose-simple.yml ps

print_status $BLUE "Following logs..."
docker-compose -f docker-compose-simple.yml logs -f
