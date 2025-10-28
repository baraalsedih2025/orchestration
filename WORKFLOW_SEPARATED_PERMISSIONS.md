# Workflow: Separated Docker Permissions

This guide is for environments where different users have different permissions:
- **Mentor/Admin**: Has Docker permissions, manages containers
- **User**: No Docker permissions, runs training code

## 🔧 Setup (One-time)

### 1. Mentor/Admin Setup Containers

**Build the Docker images:**
```bash
./admin_build_containers.sh
```

This creates 4 Docker images, one for each worker.

## 🚀 Running Training

### Step 1: Mentor Starts Containers

**Mentor runs:**
```bash
./admin_start_containers.sh
```

This will:
- Start 4 containers: ml-worker-0, ml-worker-1, ml-worker-2, ml-worker-3
- Containers stay alive but don't start training automatically
- Containers are ready for user commands

**Output:**
```
✅ All 4 containers started successfully!
Containers are running in the background (sleep infinity).
User can now run training from their account.
```

### Step 2: User Runs Training

**From your account (no Docker permissions needed):**
```bash
./run_training.sh
```

This will:
- Start training on all 4 containers
- Each worker runs distributed training code
- Training runs across all 4 GPUs

**Monitor training:**
```bash
./view_logs.sh
# Or view specific worker:
docker logs -f ml-worker-0
```

### Step 3: Stop Training (Keep Containers Running)

**User runs:**
```bash
./stop_training.sh
```

This stops training but keeps containers running (so you can restart training).

### Step 4: Mentor Stops Containers (When Done)

**Mentor runs:**
```bash
./admin_stop_containers.sh
```

This stops and removes all containers.

## 📋 Complete Workflow Example

```bash
# MENTOR SIDE (with Docker permissions)
$ ./admin_build_containers.sh      # Build images (one-time)
$ ./admin_start_containers.sh      # Start containers
✅ All 4 containers started successfully!

# USER SIDE (no Docker permissions needed)
$ ./run_training.sh                # Start training
$ ./view_logs.sh                   # Watch logs
# ... training runs ...
$ ./stop_training.sh               # Stop training

# MENTOR SIDE (cleanup)
$ ./admin_stop_containers.sh       # Stop and remove containers
```

## 📁 Scripts Overview

### Mentor/Admin Scripts (require Docker permissions)
- `admin_build_containers.sh` - Build Docker images (one-time)
- `admin_start_containers.sh` - Start 4 containers
- `admin_stop_containers.sh` - Stop and remove containers

### User Scripts (no Docker permissions needed)
- `run_training.sh` - Start training on all workers
- `stop_training.sh` - Stop training (keeps containers running)
- `view_logs.sh` - View logs from all workers

## 🔍 Architecture

```
Mentor Account (Docker permissions)
└── Manages containers lifecycle

User Account (No Docker permissions)
└── Executes training inside running containers

Containers (ml-worker-0 through ml-worker-3)
├── Each has 1 GPU
├── Environment variables configured
├── Shared volumes (data, checkpoints, logs)
└── Can be controlled by user via docker exec
```

## 🛠️ Troubleshooting

### Containers not running
```bash
# Check status
docker ps | grep ml-worker

# If not running, mentor needs to:
./admin_start_containers.sh
```

### Can't run training
```bash
# Verify containers are running
docker ps | grep ml-worker

# Check if training is already running
docker logs ml-worker-0
```

### Check GPU access
```bash
# Inside container
docker exec ml-worker-0 nvidia-smi
```

## 📊 Key Files

- `docker-compose.manual.yml` - Container definitions (sleep infinity instead of auto-start)
- `Dockerfile` - Image definition
- `slurm_training_demo.py` - Training code
- `config.json` - Configuration

## 💡 Benefits of This Approach

✅ **Separation of concerns**: Admin manages infrastructure, user runs experiments
✅ **No permission conflicts**: User doesn't need Docker group membership
✅ **Shared access**: Multiple users can run training on same containers
✅ **Easy debugging**: Containers stay alive for inspection
✅ **Flexible**: User can start/stop training without waiting for mentor

