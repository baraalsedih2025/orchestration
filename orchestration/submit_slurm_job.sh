#!/bin/bash
# SLURM Job Submission Script for Multi-Server ML Training
# =======================================================

#SBATCH --job-name=ml_training_demo
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load modules (adjust based on your cluster)
module load python/3.8
module load cuda/11.8
module load pytorch/2.1.0

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Create necessary directories
mkdir -p logs checkpoints data

# Print job information
echo "============================================================"
echo "SLURM Multi-Server ML Training Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS (4 GPUs total)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Rank: $RANK, Local Rank: $LOCAL_RANK"
echo "============================================================"

# Run the training script
./run_training.sh

echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
