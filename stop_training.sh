#!/bin/bash
# ============================================
# User Script: Stop Training
# ============================================
# This script stops training on all containers

set -e

echo "============================================================"
echo "Stopping Training on All Containers"
echo "============================================================"

# Stop python processes in each container
for worker in ml-worker-0 ml-worker-1 ml-worker-2 ml-worker-3; do
    echo "Stopping training on ${worker}..."
    docker exec ${worker} pkill -f "slurm_training_demo.py" || true
done

echo "✅ Training stopped on all containers"
echo "Containers are still running. To stop containers:"
echo "  Ask mentor to run: ./admin_stop_containers.sh"
echo "============================================================"

