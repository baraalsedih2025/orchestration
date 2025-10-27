# PowerShell Script to Run Training on Docker Containers
# ======================================================

# Colors for output
function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colorMap = @{
        "Red" = [System.ConsoleColor]::Red
        "Green" = [System.ConsoleColor]::Green
        "Yellow" = [System.ConsoleColor]::Yellow
        "Blue" = [System.ConsoleColor]::Blue
        "Cyan" = [System.ConsoleColor]::Cyan
        "White" = [System.ConsoleColor]::White
    }
    
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

Write-Host "============================================================"
Write-Host "Multi-Server ML Training Demo"
Write-Host "============================================================"

# Check if Docker is available
Write-Status "Checking Docker installation..." "Blue"
try {
    $dockerVersion = docker --version 2>$null
    if (-not $dockerVersion) {
        Write-Status "Docker is not installed or not running" "Red"
        Write-Status "Please install Docker Desktop for Windows" "Yellow"
        exit 1
    }
    Write-Status "Docker found: $dockerVersion" "Green"
} catch {
    Write-Status "Docker is not installed or not running" "Red"
    Write-Status "Please install Docker Desktop for Windows" "Yellow"
    exit 1
}

# Check if Docker daemon is running
Write-Status "Checking Docker daemon..." "Blue"
try {
    docker info | Out-Null
    Write-Status "Docker daemon is running" "Green"
} catch {
    Write-Status "Docker daemon is not running" "Red"
    Write-Status "Please start Docker Desktop" "Yellow"
    exit 1
}

# Check for docker-compose or docker compose
Write-Status "Checking Docker Compose..." "Blue"
$dockerComposeCmd = "docker-compose"
if (-not (Get-Command $dockerComposeCmd -ErrorAction SilentlyContinue)) {
    $dockerComposeCmd = "docker"
    $composeCmd = "compose"
    if (-not (Get-Command $dockerComposeCmd -ErrorAction SilentlyContinue)) {
        Write-Status "Docker Compose is not installed" "Red"
        exit 1
    }
} else {
    $composeCmd = "compose"
}
Write-Status "Using: $dockerComposeCmd $composeCmd" "Green"

# Create necessary directories
Write-Status "Creating directories..." "Blue"
if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" }
if (-not (Test-Path "checkpoints")) { New-Item -ItemType Directory -Path "checkpoints" }
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }
Write-Status "Directories ready" "Green"

# Stop any existing containers
Write-Status "Stopping existing containers if any..." "Blue"
try {
    if ($dockerComposeCmd -eq "docker-compose") {
        & docker-compose down 2>$null
    } else {
        & docker compose down 2>$null
    }
} catch {
    # Ignore errors if containers don't exist
}

# Build Docker images
Write-Status "Building Docker images for 4 workers..." "Blue"
if ($dockerComposeCmd -eq "docker-compose") {
    & docker-compose build
} else {
    & docker compose build
}
Write-Status "Docker images built successfully" "Green"

# Start Docker containers
Write-Status "Starting 4 Docker containers..." "Blue"
Write-Status "  Worker 0 (Master): ml-worker-0" "Yellow"
Write-Status "  Worker 1:         ml-worker-1" "Yellow"
Write-Status "  Worker 2:         ml-worker-2" "Yellow"
Write-Status "  Worker 3:         ml-worker-3" "Yellow"
Write-Status "  Total: 4 containers (1 GPU per container)" "Yellow"
Write-Host ""

if ($dockerComposeCmd -eq "docker-compose") {
    & docker-compose up -d
} else {
    & docker compose up -d
}

# Wait for containers to be ready
Write-Status "Waiting for containers to initialize..." "Blue"
Start-Sleep -Seconds 5

# Check if containers are running
Write-Status "Checking container status..." "Blue"
if ($dockerComposeCmd -eq "docker-compose") {
    $containers = & docker-compose ps
} else {
    $containers = & docker compose ps
}

Write-Host ""
Write-Status "Container Status:" "Blue"
$containers
Write-Host ""

# Show logs
Write-Status "============================================================" "Blue"
Write-Status "Training Logs:" "Blue"
Write-Status "============================================================" "Blue"
Write-Status "Press Ctrl+C to stop (containers will continue running)" "Yellow"
Write-Host ""

if ($dockerComposeCmd -eq "docker-compose") {
    & docker-compose logs -f
} else {
    & docker compose logs -f
}

