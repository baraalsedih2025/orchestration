#!/bin/bash
# ============================================
# User Script: View Container Logs
# ============================================
# View logs from all containers

set -e

echo "============================================================"
echo "Container Logs (Press Ctrl+C to exit)"
echo "============================================================"

docker logs -f ml-worker-0 ml-worker-1 ml-worker-2 ml-worker-3

