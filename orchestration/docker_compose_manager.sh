#!/bin/bash
# Docker Compose Management Script
# ===================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

case "${1:-}" in
    start)
        print_status $BLUE "Building Docker images for 4 workers..."
        docker-compose build
        
        print_status $BLUE "Starting 4 Docker containers..."
        docker-compose up -d
        
        print_status $GREEN "✓ All 4 workers started"
        print_status $BLUE "Following logs..."
        docker-compose logs -f
        ;;
    
    stop)
        print_status $BLUE "Stopping all containers..."
        docker-compose down
        print_status $GREEN "✓ All containers stopped"
        ;;
    
    restart)
        print_status $BLUE "Restarting containers..."
        docker-compose restart
        print_status $GREEN "✓ Containers restarted"
        ;;
    
    logs)
        print_status $BLUE "Showing logs..."
        docker-compose logs -f
        ;;
    
    status)
        print_status $BLUE "Container Status:"
        docker-compose ps
        ;;
    
    clean)
        print_status $YELLOW "Cleaning up..."
        docker-compose down -v
        print_status $GREEN "✓ Cleanup completed"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Build and start 4 Docker containers"
        echo "  stop    - Stop all containers"
        echo "  restart - Restart containers"
        echo "  logs    - Show logs from all containers"
        echo "  status  - Show container status"
        echo "  clean   - Stop and remove containers"
        ;;
esac
