# PowerShell Script to Run Training using Docker-in-Docker
# =====================================================

Write-Host "============================================================"
Write-Host "Multi-Server ML Training Demo - Docker-in-Docker"
Write-Host "============================================================"
Write-Host ""

# Check if Docker is available
Write-Host "Checking Docker..." -ForegroundColor Blue
try {
    docker version | Out-Null
    Write-Host "✓ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not available" -ForegroundColor Red
    exit 1
}

# Clean up any existing containers
Write-Host "Cleaning up..." -ForegroundColor Blue
docker ps -a -q --filter name=dind-host | ForEach-Object { docker rm -f $_ } 2>$null
Write-Host "✓ Cleanup complete" -ForegroundColor Green

# Build the host container with DinD
Write-Host "Building host container..." -ForegroundColor Blue
docker build -f Dockerfile.host -t dind-host:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

# Check if we're in WSL or native Windows
$isWSL = $false
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    $wslVersion = wsl --status 2>&1
    if ($wslVersion -match "WSL") {
        $isWSL = $true
    }
}

# Determine Docker socket path
if ($isWSL) {
    $dockerSocket = "//var/run/docker.sock:/var/run/docker.sock"
    $mountType = "bind"
} else {
    # Windows Docker Desktop
    $dockerSocket = "\\.\pipe\docker_engine:\\.\pipe\docker_engine"
    $mountType = "named-pipe"
}

Write-Host "Starting training with Docker-in-Docker..." -ForegroundColor Yellow
Write-Host "This will create a host container that runs Docker," -ForegroundColor Yellow
Write-Host "which will run 4 worker containers inside it." -ForegroundColor Yellow
Write-Host ""

# Convert Windows path to Unix-style for docker run
$currentPath = (Get-Location).Path -replace '\\', '/'
$currentPath = $currentPath -replace '^([A-Z]):', '/$1'

# Run the host container with DinD
Write-Host "Running containers..." -ForegroundColor Blue
docker run --rm --privileged `
    --name dind-host `
    -v "${currentPath}:/app" `
    -v "${currentPath}/data:/app/data" `
    -v "${currentPath}/checkpoints:/app/checkpoints" `
    -v "${currentPath}/logs:/app/logs" `
    dind-host:latest

Write-Host ""
Write-Host "✓ Training complete!" -ForegroundColor Green
Write-Host "Check the logs/ directory for training logs." -ForegroundColor Cyan

