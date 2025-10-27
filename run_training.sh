#!/bin/bash
# Simple SLURM Multi-Server ML Training Runner
# ===========================================

set -e

echo "============================================================"
echo "SLURM Multi-Server ML Training Demo"
echo "============================================================"

# Check if we're in a SLURM environment
if [ -z "$SLURM_JOB_ID" ]; then
    echo "⚠️  Not running in SLURM environment"
    echo "   This script is designed to run within a SLURM job"
    echo "   Submit with: sbatch submit_slurm_job.sh"
    echo "============================================================"
    exit 1
fi

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Rank: $RANK, Local Rank: $LOCAL_RANK"
echo "============================================================"

# Create necessary directories
mkdir -p logs checkpoints data

# Run the training script
echo "🚀 Starting training..."
python slurm_training_demo.py --config config.json

echo "============================================================"
echo "✅ Training completed!"
echo "Check checkpoints/ and logs/ directories for results"
echo "============================================================"
