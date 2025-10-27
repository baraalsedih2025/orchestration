# Start Here - Running Your Distributed Training Demo

## Current Status

Your distributed training setup is ready at: `/workspace/Bara/notebooks/orchestration`

The setup includes:
- ✅ 4 Docker containers (ml-worker-0, ml-worker-1, ml-worker-2, ml-worker-3)
- ✅ Each container gets 1 GPU
- ✅ PyTorch DDP distributed training
- ✅ ResNet50 on CIFAR-10
- ✅ Automatic checkpointing and logging

## How to Run on Your Machine

Since Docker is not available in this environment, here's how to run it on your machine:

### Option 1: Using the Provided Script (Recommended)

```bash
# Navigate to the project
cd /workspace/Bara/notebooks/orchestration

# Make the script executable (if needed)
chmod +x run_training_docker.sh

# Run the training
./run_training_docker.sh
```

This will:
1. Build the Docker images
2. Start 4 containers
3. Run distributed training
4. Show logs in real-time

### Option 2: Manual Docker Compose

```bash
cd /workspace/Bara/notebooks/orchestration

# Build the images
docker-compose build

# Start all 4 containers
docker-compose up

# Or start in background
docker-compose up -d

# Watch logs
docker-compose logs -f

# Stop when done
docker-compose down
```

## What Happens When You Run It

1. **Docker builds** the image with PyTorch, CUDA, and your training code
2. **4 containers start**:
   - ml-worker-0 (Master, Rank 0) - GPU 0
   - ml-worker-1 (Rank 1) - GPU 1
   - ml-worker-2 (Rank 2) - GPU 2
   - ml-worker-3 (Rank 3) - GPU 3

3. **Training initializes**:
   - Download CIFAR-10 dataset (first time only)
   - Initialize PyTorch DDP across 4 workers
   - Configure ResNet50 model

4. **Training runs**:
   - 100 epochs total
   - Distributed data sampling across workers
   - Gradient synchronization
   - Distributed validation

5. **Outputs**:
   - Checkpoints: `./checkpoints/checkpoint_epoch_*.pth`
   - Logs: `./logs/`
   - Dataset: `./data/`

## Expected Output

You'll see logs like:
```
============================================================
SLURM Multi-Server ML Training Demo
============================================================
Job ID: docker-job-001
Nodes: ml-worker-0,ml-worker-1,ml-worker-2,ml-worker-3
Master: ml-worker-0:12355
============================================================

🚀 Starting training...
✓ Loaded configuration from config.json
✓ Distributed training initialized on rank 0 using nccl
✓ Model setup complete on rank 0 - resnet50
✓ Data loaders setup complete on rank 0

Epoch: 0, Batch: 0, Loss: 2.3012, Acc: 10.94%
Epoch: 0, Batch: 100, Loss: 1.8765, Acc: 29.94%
📊 Epoch 0 - Time: 47.23s - Train Acc: 29.94%, Val Acc: 43.40%
✓ Checkpoint saved: checkpoints/checkpoint_epoch_0.pth
```

## Prerequisites

You need these installed on your machine:

### 1. Docker & Docker Compose
```bash
# Check if installed
docker --version
docker-compose --version

# Install if needed (Ubuntu)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
```

### 2. NVIDIA Container Toolkit
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### 3. At Least 4 GPUs
```bash
# Check available GPUs
nvidia-smi --list-gpus

# You need at least 4 GPUs for this demo
```

## Customization

Edit `config.json` to change:
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size per worker
- `learning_rate`: Learning rate
- `model.name`: Model architecture
- `data.dataset`: Dataset name

## Monitor Training

```bash
# View all logs
docker-compose logs -f

# View specific worker
docker logs -f ml-worker-0

# Check container status
docker-compose ps

# Check GPU usage
watch -n 1 nvidia-smi
```

## Stop Training

```bash
# Graceful shutdown
docker-compose down

# Force stop
docker-compose kill
```

## Troubleshooting

### No GPUs detected
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# Verify
nvidia-smi
```

### Containers can't communicate
```bash
# Check network
docker network inspect orchestration_ml-network

# Test connectivity
docker exec ml-worker-0 ping -c 2 ml-worker-1
```

### Out of memory
Reduce batch size in `config.json`:
```json
"batch_size": 16  // Instead of 32
```

Or use fewer workers by editing `docker-compose.yml` and setting `WORLD_SIZE=2`.

## Files Reference

- `docker-compose.yml` - Container orchestration (4 workers, GPU allocation)
- `Dockerfile` - Container image with PyTorch, CUDA
- `slurm_training_demo.py` - Training script with DDP
- `config.json` - All training parameters
- `requirements.txt` - Python dependencies
- `run_training_docker.sh` - Easy start script
- `HOW_TO_DEMO.md` - Detailed guide
- `QUICK_START.txt` - Quick reference

## Architecture

```
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ ml-worker-0   │  │ ml-worker-1   │  │ ml-worker-2   │  │ ml-worker-3   │
│ (Master)      │  │               │  │               │  │               │
│ GPU 0         │  │ GPU 1         │  │ GPU 2         │  │ GPU 3         │
│ Rank 0        │  │ Rank 1        │  │ Rank 2        │  │ Rank 3        │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                   │                   │                   │
        └───────────────────┼───────────────────┼───────────────────┘
                            │                   │
                    ┌───────┴───────────────────┴────────┐
                    │  PyTorch DDP (NCCL backend)        │
                    │  - All-reduce gradients            │
                    │  - Distributed data sampling       │
                    │  - Synchronized validation         │
                    └────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │  Shared Storage (Volume Mounts)     │
        │  - ./data (dataset)                  │
        │  - ./checkpoints (models)            │
        │  - ./logs (training logs)            │
        └─────────────────────────────────────┘
```

## Next Steps After Running

1. Check generated checkpoints in `./checkpoints/`
2. Review training logs in `./logs/`
3. Experiment with different models or hyperparameters
4. Scale to more workers on a real cluster
5. Try different datasets

## Alternative: Run Without Docker (Single Worker)

If you want to test the code without Docker, you can modify the script to run a single process:

```bash
cd /workspace/Bara/notebooks/orchestration

# Set environment variables for single worker
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

# Run training (modify script to skip SLURM env vars)
python3 -c "
import json
import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
import slurm_training_demo
config = json.load(open('config.json'))
# Need to modify trainer to work without SLURM
"
```

---

**Ready to run?** Execute: `./run_training_docker.sh` on your machine with Docker and GPUs installed!

