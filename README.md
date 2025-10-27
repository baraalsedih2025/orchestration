# Distributed ML Training - 4 Docker Containers

Run distributed training across **4 Docker containers**, each acting as a separate machine with its own GPU, using PyTorch DDP (Distributed Data Parallel).

## 🎯 What This Does

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Container 1    │    │  Container 2    │    │  Container 3    │    │  Container 4    │
│   GPU 0         │    │   GPU 1         │    │   GPU 2         │    │   GPU 3         │
│   Rank 0        │    │   Rank 1        │    │   Rank 2        │    │   Rank 3        │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │                      │
         └──────────────────────┼──────────────────────┼──────────────────────┘
                               │                      │
                    ┌──────────▼──────────────────────▼──────────┐
                    │      PyTorch DDP Training                   │
                    │  - 4 GPUs total                             │
                    │  - Distributed data sampling                │
                    │  - Gradient synchronization                 │
                    │  - ResNet50 on CIFAR-10                     │
                    └────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4 GPUs (1 per container)
- NVIDIA Container Toolkit

### Run Training
```bash
./run_training_docker.sh
```

This will:
1. Build Docker images
2. Start 4 containers (ml-worker-0, ml-worker-1, ml-worker-2, ml-worker-3)
3. Run distributed training with PyTorch DDP
4. Show logs from all workers

### Manual Run
```bash
# Build and start
docker-compose up --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📊 Configuration

Edit `config.json` to change:
- Model: `resnet50`, `simplecnn`
- Dataset: `cifar10`
- Training: epochs, batch size, learning rate
- Distributed: number of workers (currently 4)

## 📁 Project Structure

```
orchestration/
├── docker-compose.yml      # Defines 4 containers
├── Dockerfile              # Container image with PyTorch/CUDA
├── slurm_training_demo.py  # Training code with DDP
├── config.json            # All training parameters
├── requirements.txt       # Python dependencies
└── run_training_docker.sh # One-command runner
```

## 🧠 Architecture

- **4 Docker Containers**: Each container acts as an independent machine
- **PyTorch DDP**: Distributed Data Parallel across containers
- **NCCL Backend**: GPU communication between containers
- **ResNet50**: Model architecture (configurable)
- **CIFAR-10**: Dataset (downloads automatically)

## 📝 Training Process

1. Each container initializes as a worker (ranks 0-3)
2. Worker 0 becomes the master
3. Data is distributed across workers
4. Model is replicated on each GPU
5. Gradients are synchronized via all-reduce
6. Validation and checkpoints are managed

## 🔧 Key Files

- **`docker-compose.yml`**: Orchestrates 4 containers with GPU allocation
- **`Dockerfile`**: Builds image with PyTorch, CUDA, dependencies
- **`slurm_training_demo.py`**: Training loop with DDP initialization
- **`config.json`**: All hyperparameters and settings
- **`run_training_docker.sh`**: Convenience script to run everything

## 🛠️ Monitor Training

```bash
# Check GPU usage
nvidia-smi

# View specific worker logs
docker logs -f ml-worker-0

# Check all containers
docker-compose ps
```

## 🎯 Expected Output

```
Epoch: 0, Batch: 0, Loss: 2.3012, Acc: 10.94%
Epoch: 0, Batch: 100, Loss: 1.8765, Acc: 29.94%
📊 Epoch 0 - Train Acc: 29.94%, Val Acc: 43.40%
✓ Checkpoint saved: checkpoints/checkpoint_epoch_0.pth
```

## 🔍 Troubleshooting

**No GPUs detected:**
```bash
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

**Containers can't communicate:**
```bash
docker network inspect orchestration_ml-network
```

**Out of memory:**
Reduce batch size in `config.json`: `"batch_size": 16`
