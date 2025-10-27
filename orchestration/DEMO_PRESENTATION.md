# Running the 4 Docker ML Training Demo

## Overview
This demo simulates a 4-machine cluster using Docker containers, demonstrating how to train ML models across multiple machines using PyTorch DDP (Distributed Data Parallel).

## Setup
- **Current**: 1 machine with 4 GPUs → 4 Docker containers (1 GPU each)
- **Future**: 4 machines with 4 GPUs each → 16 GPUs total
- **Method**: PyTorch DDP for distributed training

## Quick Start

### Prerequisites
```bash
# Install Docker and Docker Compose
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Start Docker daemon
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
```

### Run the Demo
```bash
cd /workspace/Bara/Notebooks/orchestration

# One command to build and start 4 Docker workers
./run_training_docker.sh
```

### What Happens
1. **Builds Docker images** for 4 workers
2. **Starts 4 containers** (ml-worker-0, ml-worker-1, ml-worker-2, ml-worker-3)
3. **Connects via Docker network** (172.20.0.0/16)
4. **Initializes PyTorch DDP** across all 4 workers
5. **Trains ResNet50** on CIFAR-10 using distributed training
6. **Synchronizes gradients** across all workers
7. **Saves checkpoints** (rank 0 only)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ml-worker-0     │    │ ml-worker-1     │    │ ml-worker-2     │    │ ml-worker-3     │
│ (Master)        │    │                 │    │                 │    │                 │
│                 │    │                 │    │                 │    │                 │
│ GPU 0           │    │ GPU 1           │    │ GPU 2           │    │ GPU 3           │
│ ResNet50 + DDP  │    │ ResNet50 + DDP  │    │ ResNet50 + DDP  │    │ ResNet50 + DDP  │
│ Rank: 0         │    │ Rank: 1         │    │ Rank: 2         │    │ Rank: 3         │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                    ┌─────────────▼─────────────┐       │
                    │  PyTorch DDP Backend      │       │
                    │  (Gloo/NCCL)              │       │
                    └─────────────────────────┘       │
                                                         │
                    ┌─────────────────────────────────────▼─────────────────────────────────────┐
                    │                    Distributed Training Coordination                        │
                    │  - Gradient Synchronization (all_reduce)                                  │
                    │  - Model Parameter Averaging                                             │
                    │  - Distributed Data Sampling                                              │
                    │  - Checkpoint Management (rank 0)                                        │
                    └─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

All parameters are in `config.json`:

### Current Setup (4 GPUs)
```json
{
    "distributed": {
        "backend": "nccl",
        "world_size": 4
    },
    "slurm": {
        "nodes": 4,
        "ntasks_per_node": 1,
        "gres": "gpu:1"
    }
}
```

### Future Setup (16 GPUs across 4 machines)
```json
{
    "distributed": {
        "backend": "nccl",
        "world_size": 16
    },
    "slurm": {
        "nodes": 4,
        "ntasks_per_node": 4,
        "gres": "gpu:4"
    }
}
```

## Key Files

- `slurm_training_demo.py` - Main training script using DDP
- `docker-compose.yml` - 4 worker configuration  
- `run_training_docker.sh` - One command to run everything
- `config.json` - All configuration parameters
- `Dockerfile` - Docker image definition

## Monitoring

```bash
# View logs from all workers
docker logs ml-worker-0 -f  # Master worker
docker logs ml-worker-1 -f  # Worker 1
docker logs ml-worker-2 -f  # Worker 2  
docker logs ml-worker-3 -f  # Worker 3

# Check GPU usage
watch -n 1 nvidia-smi

# Check container status
docker ps
docker stats
```

## Demo Presentation Script

```bash
#!/bin/bash
# Demo Presentation Script

echo "============================================================"
echo "Multi-Machine ML Training Demo"
echo "============================================================"
echo ""
echo "Current Setup:"
echo "  - 1 physical machine"
echo "  - 4 Docker containers (simulating 4 machines)"
echo "  - 4 GPUs total (1 per container)"
echo ""
echo "Future Setup:"
echo "  - 4 physical machines"
echo "  - 16 GPUs total (4 per machine)"
echo ""
echo "Technology:"
echo "  - PyTorch DDP (Distributed Data Parallel)"
echo "  - Docker containers for isolation"
echo "  - SLURM orchestration (production)"
echo ""
echo "Starting demo in 5 seconds..."
sleep 5

# Run the demo
./run_training_docker.sh
```

## Scaling to 4 Machines

When you have 4 physical machines:

1. **Deploy on each machine**:
   ```bash
   scp -r orchestration/ user@machine1:/path/
   scp -r orchestration/ user@machine2:/path/
   scp -r orchestration/ user@machine3:/path/
   scp -r orchestration/ user@machine4:/path/
   ```

2. **Update network configuration**:
   - Update `docker-compose.yml` with actual machine IPs
   - Or use SLURM for orchestration

3. **Run with SLURM**:
   ```bash
   sbatch submit_slurm_job.sh
   ```

## Key Concepts Demonstrated

✅ **Distributed Training**: PyTorch DDP across multiple workers  
✅ **Gradient Synchronization**: All-reduce operations  
✅ **Data Sharding**: DistributedSampler splits dataset  
✅ **Model Parallelism**: Each worker has a replica  
✅ **Checkpoint Management**: Only rank 0 saves  
✅ **Fault Tolerance**: Configurable for failures  
✅ **Scalability**: Easy to scale from 4 to 16 GPUs

---

**Ready to present to your team!** 🚀

