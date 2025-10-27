# How to Run Distributed ML Training Demo

## Quick Start

You have a complete setup for distributed ML training across 4 Docker containers. Each container will use 1 GPU.

### Prerequisites

1. **Docker & Docker Compose** - Make sure Docker is installed and running
2. **NVIDIA GPU Support** - Your system needs NVIDIA Container Toolkit installed
3. **At least 4 GPUs** - You need 4 GPUs for this demo (one per container)

### Step 1: Install NVIDIA Container Toolkit (if not already installed)

```bash
# For Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 2: Verify GPU Access

```bash
# Check that GPUs are accessible
nvidia-smi

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### Step 3: Navigate to the Project Directory

```bash
cd /workspace/Bara/notebooks/orchestration
```

### Step 4: Run the Demo

There are two ways to run:

#### Option A: Simple Script (Recommended)

```bash
# Make the script executable
chmod +x run_training_docker.sh

# Run the training
./run_training_docker.sh
```

#### Option B: Manual Docker Compose Commands

```bash
# Build the Docker images
docker-compose build

# Start all 4 containers
docker-compose up -d

# Watch the logs
docker-compose logs -f
```

### Step 5: Monitor Training

Watch the logs to see training progress:

```bash
# Follow all logs
docker-compose logs -f

# Follow specific worker logs
docker logs -f ml-worker-0
docker logs -f ml-worker-1
docker logs -f ml-worker-2
docker logs -f ml-worker-3

# Check container status
docker-compose ps
```

### Step 6: Stop Training

When you're done:

```bash
# Stop all containers
docker-compose down

# Or force stop
docker-compose kill
```

## What Happens During Training

1. **4 Docker containers start** - Each acts as a "server" with 1 GPU
   - ml-worker-0 (Master/Rank 0)
   - ml-worker-1 (Rank 1)  
   - ml-worker-2 (Rank 2)
   - ml-worker-3 (Rank 3)

2. **Distributed Training Initializes** - PyTorch DDP connects all 4 workers

3. **CIFAR-10 Dataset** - Automatically downloaded to `./data/`

4. **Training Runs** - ResNet50 model trains across 4 GPUs with:
   - Distributed data sampling
   - Gradient synchronization
   - Distributed validation

5. **Checkpoints Saved** - Models saved to `./checkpoints/`

6. **Logs** - Training logs saved to `./logs/`

## Configuration

All training parameters can be adjusted in `config.json`:

- Model: ResNet50
- Dataset: CIFAR-10
- Batch size: 32
- Learning rate: 0.1
- Number of epochs: 100
- Workers: 4 (1 GPU per worker)

## Architecture

```
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ ml-worker-0   │  │ ml-worker-1   │  │ ml-worker-2   │  │ ml-worker-3   │
│ GPU 0         │  │ GPU 1         │  │ GPU 2         │  │ GPU 3         │
│ (Master)      │  │               │  │               │  │               │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                   │                   │                   │
        └───────────────────┼───────────────────┼───────────────────┘
                            │                   │
                    ┌───────┴───────────────────┴────────┐
                    │  PyTorch DDP Communication        │
                    │  - Gradient Synchronization        │
                    │  - All-reduce Operations          │
                    └────────────────────────────────────┘
```

## Troubleshooting

### Issue: GPUs not detected in containers

```bash
# Check GPUs on host
nvidia-smi

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

# If no GPUs appear, reinstall NVIDIA Container Toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: Containers fail to start

```bash
# Check Docker version (needs >= 19.03)
docker --version

# Check compose version
docker-compose --version

# View detailed logs
docker-compose logs
```

### Issue: Connection refused between workers

```bash
# Check if containers are on the same network
docker network inspect orchestration_ml-network

# Check if containers can reach each other
docker exec ml-worker-0 ping -c 2 ml-worker-1
docker exec ml-worker-1 ping -c 2 ml-worker-0
```

### Issue: Out of memory

If you don't have 4 GPUs or want to use fewer containers:

1. Edit `config.json` and change `world_size` to match your number of GPUs
2. Edit `docker-compose.yml` to remove unneeded workers
3. Or use single-container mode (see below)

## Single-Container Demo (If You Have Only 1-3 GPUs)

If you have fewer than 4 GPUs, you can modify the setup:

### Option 1: Reduce Container Count

Edit `docker-compose.yml` and remove workers 2 and 3, then:
- Set `WORLD_SIZE=2` (or however many workers you keep)
- Update `SLURM_NTASKS` accordingly

### Option 2: Use Single Worker Mode

```bash
# Run a single container with all GPUs visible
docker run -it --rm \
  --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/checkpoints:/app/checkpoints \
  -v ${PWD}/logs:/app/logs \
  -e RANK=0 \
  -e WORLD_SIZE=1 \
  orchestration-ml-worker-0 \
  python3 -c "import slurm_training_demo; import json; config = json.load(open('config.json')); trainer = slurm_training_demo.SLURMMultiServerTrainer(config); trainer.train()"
```

## Expected Output

When training starts, you should see:

```
============================================================
SLURM Multi-Server ML Training Demo
============================================================
Job ID: docker-job-001
Job Name: ml_training_demo
Nodes: ml-worker-0,ml-worker-1,ml-worker-2,ml-worker-3
Tasks: 4
Master: ml-worker-0:12355
Rank: 0, Local Rank: 0
============================================================

🚀 Starting training...
✓ Loaded configuration from config.json
✓ Distributed training initialized on rank 0 using nccl
✓ Model setup complete on rank 0 - resnet50
✓ Data loaders setup complete on rank 0
  Dataset: cifar10, Batch size: 32, Workers: 4

Epoch: 0, Batch: 0, Loss: 2.3012, Acc: 10.94%
Epoch: 0, Batch: 100, Loss: 1.8765, Acc: 29.94%
📊 Epoch 0 - Time: 47.23s - Train Acc: 29.94%, Val Acc: 43.40%
✓ Checkpoint saved: checkpoints/checkpoint_epoch_0.pth

Epoch: 1, Batch: 0, Loss: 1.6543, Acc: 35.21%
...
```

## Performance Tips

1. **Increase batch size** if you have more memory (edit `config.json`)
2. **Use more workers** if you have more GPUs  
3. **Enable mixed precision** training for faster training (advanced)

## Next Steps

After the demo works:
1. Experiment with different models (edit `config.json`)
2. Try different datasets
3. Tune hyperparameters
4. Scale to more workers on a real cluster

## Files Structure

```
orchestration/
├── docker-compose.yml          # 4-container setup
├── Dockerfile                  # Container image
├── slurm_training_demo.py      # Training script
├── config.json                 # All parameters
├── requirements.txt            # Python dependencies
├── run_training_docker.sh     # Easy start script
├── data/                       # Dataset storage
├── checkpoints/                # Saved models
└── logs/                       # Training logs
```

## Support

If you encounter issues:
1. Check container logs: `docker logs ml-worker-0`
2. Verify GPU access: `nvidia-smi` and `docker run --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`
3. Check network connectivity between containers
4. Ensure NVIDIA Container Toolkit is properly installed

