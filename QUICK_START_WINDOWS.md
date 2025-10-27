# Quick Start Guide for Windows

## Prerequisites

1. **Install Docker Desktop** from: https://www.docker.com/products/docker-desktop/
2. **Ensure Docker Desktop is running** (check system tray for Docker icon)
3. **Install NVIDIA Container Toolkit** (if you have NVIDIA GPUs)

## Step-by-Step Instructions

### Step 1: Start Docker Desktop

1. Open Docker Desktop application
2. Wait until it shows "Docker Desktop is running" in the system tray
3. You should see a green icon with a whale

### Step 2: Open PowerShell

1. Press `Windows Key + X`
2. Select "Windows PowerShell" or "Terminal"
3. Navigate to the orchestration folder:
   ```powershell
   cd C:\Users\BaraAlsedih\Downloads\orchestration
   ```

### Step 3: Verify Docker is Running

```powershell
docker info
```

If you see Docker information, you're good to go!

### Step 4: Run Training

**Option A - PowerShell Script (Recommended):**
```powershell
.\run_training_docker.ps1
```

**Option B - Batch File:**
```cmd
.\run_training_docker.bat
```

**Option C - Direct Docker Commands:**
```powershell
# Build and start containers
docker-compose build
docker-compose up

# View logs
docker-compose logs -f

# Stop containers (in another terminal)
docker-compose down
```

### Step 5: Monitor Training

Training logs will appear in real-time. You'll see:
- Workers initializing
- Data loading progress
- Training metrics (loss, accuracy)
- Checkpoint saves

## Troubleshooting

### "Docker is not running" error

**Solution:** Start Docker Desktop application first, then wait 30 seconds and try again.

### "PowerShell script execution is disabled"

**Solution:** Run this command:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Cannot connect to Docker daemon"

**Solution:** 
1. Restart Docker Desktop
2. Wait for it to fully start
3. Try again

### "No space left on device"

**Solution:** Clean up Docker:
```powershell
docker system prune -a
```

### Containers won't start

**Solution:**
```powershell
# Check logs
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## What Happens During Training

1. **Build Phase** (first time only):
   - Downloads PyTorch Docker image
   - Installs dependencies
   - Builds custom image
   - Takes 5-10 minutes

2. **Startup Phase**:
   - Starts 4 containers (ml-worker-0 through ml-worker-3)
   - Each container gets 1 GPU
   - Downloads CIFAR-10 dataset (first time)

3. **Training Phase**:
   - Runs distributed training across 4 GPUs
   - Saves checkpoints every epoch
   - Shows training and validation accuracy

4. **Results**:
   - Checkpoints saved to `./checkpoints/`
   - Logs saved to `./logs/`
   - Model trained for 3 epochs by default

## Stop Training

Press `Ctrl+C` to stop viewing logs (containers keep running), or:

```powershell
# Stop all containers
docker-compose down

# Stop and remove data
docker-compose down -v
```

## View Results

```powershell
# View saved checkpoints
dir checkpoints

# View log files
dir logs
Get-Content logs\training_*.log
```

## That's it!

Your training will start automatically. Just make sure Docker Desktop is running first!

