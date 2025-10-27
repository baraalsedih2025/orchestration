#!/bin/bash
# Simple Local Testing Script (Single Worker)
# ===========================================

set -e

echo "============================================================"
echo "Local ML Training Demo (Single Worker Mode)"
echo "============================================================"

echo "⚠️  Running in single worker mode for local testing"
echo "   This demonstrates the training code without distributed setup"
echo "============================================================"

# Set up single worker environment
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE="1"
export RANK="0"
export LOCAL_RANK="0"
export SLURM_PROCID="0"
export SLURM_LOCALID="0"
export SLURM_NTASKS="1"
export SLURM_NODEID="0"
export SLURM_JOB_ID="test-job-001"
export SLURM_JOB_NAME="ml_training_demo"
export SLURM_JOB_NODELIST="localhost"

echo "Job ID: $SLURM_JOB_ID (test mode)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS (single worker)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Rank: $RANK, Local Rank: $LOCAL_RANK"
echo "============================================================"

# Create necessary directories
mkdir -p logs checkpoints data

echo "🚀 Starting training in single worker mode..."
echo "   Note: This runs without distributed training"
echo "   For distributed training, use SLURM cluster"
echo "============================================================"

# Run the training script
python3 slurm_training_demo.py --config config.json

echo "============================================================"
echo "✅ Training completed!"
echo "Check checkpoints/ and logs/ directories for results"
echo "============================================================"
