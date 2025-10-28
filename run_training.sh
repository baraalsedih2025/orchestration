#!/bin/bash
# ============================================
# User Script: Run Training in Containers
# ============================================
# This script runs training on all 4 containers
# Run this from your user account (no docker permissions needed)

set -e

echo "============================================================"
echo "Starting Distributed Training on 4 Containers"
echo "============================================================"

# Function to run training on a container
run_training_on_container() {
    local container_name=$1
    echo "Starting training on ${container_name}..."
    
    docker exec -d ${container_name} python3 slurm_training_demo.py --config config.json
    
    echo "✓ Training started on ${container_name}"
}

# Check if containers are running
if ! docker ps | grep -q "ml-worker-0"; then
    echo "❌ Error: Containers are not running!"
    echo "Please ask your mentor to run: ./admin_start_containers.sh"
    exit 1
fi

echo "Containers detected:"
docker ps | grep "ml-worker"

echo ""
echo "Starting training on all 4 workers..."
echo ""

# Start training on each worker (worker 0 starts immediately, others with delay)
run_training_on_container "ml-worker-0"

# Worker 1-3 start after a 5 second delay
for worker in ml-worker-1 ml-worker-2 ml-worker-3; do
    echo "Waiting 5 seconds before starting ${worker}..."
    sleep 5
    run_training_on_container "${worker}"
done

echo ""
echo "✅ Training started on all 4 containers!"
echo ""
echo "To view logs:"
echo "  docker logs -f ml-worker-0    # Master worker"
echo "  docker logs -f ml-worker-1    # Worker 1"
echo ""
echo "To stop training:"
echo "  ./stop_training.sh"
echo "============================================================"

