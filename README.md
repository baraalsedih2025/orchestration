# SLURM Multi-Server ML Training Demo

A simple demonstration of distributed machine learning training using SLURM orchestration across 4 Docker containers, with all configuration loaded from `config.json`.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Server 1      │    │   Server 2      │    │   Server 3      │    │   Server 4      │
│   ┌───────────┐  │    │   ┌───────────┐  │    │   ┌───────────┐  │    │   ┌───────────┐  │
│   │ GPU 0     │  │    │   │ GPU 0     │  │    │   │ GPU 0     │  │    │   │ GPU 0     │  │
│   │ ResNet50  │  │    │   │ ResNet50  │  │    │   │ ResNet50  │  │    │   │ ResNet50  │  │
│   │ + DDP     │  │    │   │ + DDP     │  │    │   │ + DDP     │  │    │   │ + DDP     │  │
│   └───────────┘  │    │   └───────────┘  │    │   └───────────┘  │    │   └───────────┘  │
│   Docker         │    │   Docker         │    │   Docker         │    │   Docker         │
│   Container      │    │   Container      │    │   Container      │    │   Container      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                    ┌─────────────▼─────────────┐       │
                    │      SLURM Scheduler      │       │
                    │    (Job Orchestration)    │       │
                    └───────────────────────────┘       │
                                                         │
                    ┌─────────────────────────────────────▼─────────────────────────────────────┐
                    │                    PyTorch DDP Distributed Training                        │
                    │  - 4 GPUs total (1 per server)                                         │
                    │  - Gradient synchronization across all workers                           │
                    │  - Distributed data sampling                                            │
                    │  - Checkpoint management (rank 0 only)                                  │
                    └─────────────────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
orchestration/
├── slurm_training_demo.py      # Main training script
├── Dockerfile                  # Docker image definition
├── submit_slurm_job.sh         # SLURM job submission
├── run_training.sh             # Simple training runner
├── config.json                 # All configuration parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── checkpoints/                # Model checkpoints
├── data/                       # Dataset storage
└── logs/                       # Training logs
```

## 🚀 Quick Start

### 1. Install Dependencies

**On Linux/Mac:**
```bash
cd /workspace/Bara/Notebooks/orchestration
pip install -r requirements.txt
```

**On Windows:**
```powershell
cd C:\Users\YourUser\Downloads\orchestration
pip install -r requirements.txt
```

### 2. Prerequisites

- Docker Desktop installed and running
- NVIDIA GPU(s) with CUDA support (4 GPUs recommended)
- At least 4 GPUs for the default 4-worker setup

### 3. Run Training (Choose one method)

#### Option A: Docker Compose (Local 4 Workers)

**On Linux/Mac:**
```bash
# Simple one-command start (recommended)
./run_training_docker.sh

# Or use the management script
./docker_compose_manager.sh start
```

**On Windows (PowerShell):**
```powershell
# Run the PowerShell script
.\run_training_docker.ps1

# Or manually using docker-compose
docker-compose up --build
docker-compose logs -f
```

#### Option B: SLURM (Cluster)

```bash
# Submit SLURM job
sbatch submit_slurm_job.sh
```

#### Option C: Local Testing (Single Worker)

```bash
# Test on local machine without Docker
./test_single.sh
```

### 3. Monitor Training

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/slurm_<JOB_ID>.out
```

## 🔧 Configuration

All parameters are loaded from `config.json`:

### Model & Training
```json
{
    "training": {
        "model": {
            "name": "resnet50",
            "num_classes": 10,
            "pretrained": false
        },
        "data": {
            "dataset": "cifar10",
            "batch_size": 32,
            "num_workers": 4
        },
        "optimizer": {
            "type": "sgd",
            "learning_rate": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4
        },
        "training": {
            "num_epochs": 100,
            "save_frequency": 10
        }
    }
}
```

### SLURM Resources
```json
{
    "slurm": {
        "job_name": "ml_training_demo",
        "partition": "gpu",
        "nodes": 4,
        "ntasks_per_node": 1,
        "gres": "gpu:1",
        "cpus_per_task": 8,
        "mem": "64G",
        "time": "02:00:00"
    }
}
```

### Distributed Training
```json
{
    "distributed": {
        "backend": "nccl",
        "world_size": 4
    }
}
```

## 📊 Training Process

1. **SLURM Allocation**: 4 nodes × 1 GPU = 4 total GPUs
2. **Docker Startup**: Each node runs Docker container
3. **DDP Initialization**: PyTorch DDP across all containers
4. **Data Distribution**: CIFAR-10 split across workers
5. **Model Training**: ResNet50 with gradient sync
6. **Checkpointing**: Model saved by rank 0

## 📈 Example Output

```
============================================================
SLURM Multi-Server ML Training Demo
============================================================
Job ID: 12345
Job Name: ml_training_demo
Nodes: ml-server-[1-4]
Tasks: 16
Master: ml-server-1:12355
Rank: 0, Local Rank: 0
============================================================

🚀 Starting training...
✓ Loaded configuration from config.json
✓ Distributed training initialized on rank 0 using nccl
✓ Model setup complete on rank 0 - resnet50

Epoch: 0, Batch: 0, Loss: 2.3012, Acc: 10.94%
Epoch: 0, Batch: 100, Loss: 1.8765, Acc: 29.94%
📊 Epoch 0 - Time: 47.23s - Train Acc: 29.94%, Val Acc: 43.40%
✓ Checkpoint saved: checkpoints/checkpoint_epoch_0.pth

...

✅ Training completed!
Check checkpoints/ and logs/ directories for results
============================================================
```

## 🧪 Local Testing

For environments without SLURM, use the local testing script:

```bash
# Run single worker test (no distributed training)
./test_single.sh
```

This script:
- Runs training without distributed setup
- Uses single worker mode
- Perfect for testing the training code
- No network or multi-process dependencies

**Note**: For true distributed training across 4 workers, use SLURM cluster with `sbatch submit_slurm_job.sh`.

## 🛠️ Management

### Docker Compose Commands

```bash
# Start 4 Docker containers
./docker_compose_manager.sh start

# Stop containers
./docker_compose_manager.sh stop

# View logs
./docker_compose_manager.sh logs

# Check status
./docker_compose_manager.sh status

# Restart containers
./docker_compose_manager.sh restart

# Clean up
./docker_compose_manager.sh clean
```

### SLURM Commands

```bash
# Submit job
sbatch submit_slurm_job.sh

# Check status
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# View details
scontrol show job <JOB_ID>
```

### Docker Commands

```bash
# Build image
docker build -t ml-training .

# Run container (testing)
docker run --gpus all -it ml-training
```

## 🔍 Troubleshooting

### Windows-Specific Issues

**Issue: GPUs not detected in Docker containers**
- Solution: Install NVIDIA Container Toolkit for Windows
- Check GPU availability: `nvidia-smi` in your host
- Verify Docker Desktop has GPU support enabled in Settings

**Issue: PowerShell script doesn't run**
- Solution: Set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Or run: `powershell -ExecutionPolicy Bypass -File .\run_training_docker.ps1`

**Issue: Containers can't communicate**
- Check firewall settings
- Verify containers are on the same network: `docker network inspect orchestration_ml-network`

**Issue: Permission denied errors**
- Run PowerShell as Administrator
- Or fix folder permissions: Right-click folder → Properties → Security

### Common Issues

```bash
# Check SLURM status
sinfo -p gpu

# Check job details
scontrol show job <JOB_ID>

# Check GPU allocation
nvidia-smi

# Enable debug logging
export NCCL_DEBUG=INFO  # Linux/Mac
set NCCL_DEBUG=INFO     # Windows
```

### Docker Troubleshooting

```bash
# Check if containers are running
docker ps -a

# View container logs
docker logs ml-worker-0
docker logs ml-worker-1

# Restart failed container
docker restart ml-worker-0

# Rebuild images
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## 🔄 Customization

Edit `config.json` to change:

- **Model**: `"name": "resnet50"` or `"simplecnn"`
- **Dataset**: `"dataset": "cifar10"` or `"imagenet"`
- **Resources**: `"nodes": 4`, `"gres": "gpu:1"`
- **Training**: `"num_epochs": 100`, `"batch_size": 32`

## 📚 Key Features

- ✅ **Simple Setup**: One script to run everything
- ✅ **SLURM Integration**: Professional job scheduling
- ✅ **Docker Containers**: Isolated environments
- ✅ **Config-Driven**: All parameters from config.json
- ✅ **PyTorch DDP**: Distributed training across 4 GPUs
- ✅ **ResNet50**: Production-ready model
- ✅ **Automatic Checkpointing**: Model saving

---

**Usage**: `sbatch submit_slurm_job.sh` - That's it!