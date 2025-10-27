#!/bin/bash
# Run 4 Workers with Docker Run (No Network Creation)
# ====================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${1}${2}${NC}"
}

echo "============================================================"
echo "Multi-Server ML Training Demo"
echo "Using docker run (no network creation)"
echo "============================================================"

# Create directories
mkdir -p data checkpoints logs

# Stop any existing containers
print_status $BLUE "Stopping existing containers..."
docker rm -f ml-worker-0 ml-worker-1 ml-worker-2 ml-worker-3 2>/dev/null || true

# Pull PyTorch image
print_status $BLUE "Pulling PyTorch image..."
docker pull pytorch/pytorch:latest

# Start worker 0 (master)
print_status $BLUE "Starting Worker 0..."
docker run -d \
  --name ml-worker-0 \
  --hostname ml-worker-0 \
  -e MASTER_ADDR=localhost \
  -e MASTER_PORT=12355 \
  -e WORLD_SIZE=1 \
  -e RANK=0 \
  -e LOCAL_RANK=0 \
  -e SLURM_PROCID=0 \
  -e SLURM_LOCALID=0 \
  -e SLURM_NTASKS=1 \
  -e SLURM_NODEID=0 \
  -e SLURM_JOB_ID=001 \
  -e SLURM_JOB_NAME=demo \
  -e SLURM_JOB_NODELIST=localhost \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/slurm_training_demo.py:/app/slurm_training_demo.py \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/requirements.txt:/app/requirements.txt \
  pytorch/pytorch:latest \
  bash -c "pip install -r requirements.txt && python3 slurm_training_demo.py --config config.json"

print_status $GREEN "Worker 0 started"
print_status $BLUE "Following logs..."
docker logs -f ml-worker-0
