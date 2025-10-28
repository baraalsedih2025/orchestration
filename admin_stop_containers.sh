#!/bin/bash
# ============================================
# Mentor/Admin Script: Stop Docker Containers
# ============================================
# This script stops all 4 Docker containers
# Run this with docker permissions (mentor/admin only)

set -e

echo "============================================================"
echo "Stopping Docker Containers"
echo "============================================================"

# Stop and remove containers
docker-compose -f docker-compose.manual.yml down

echo "✅ All containers stopped and removed"
echo "============================================================"

