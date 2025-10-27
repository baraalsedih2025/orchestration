@echo off
REM Batch script to run training on Docker containers
REM =================================================

echo ============================================================
echo Multi-Server ML Training Demo
echo ============================================================
echo.

REM Check if Docker is available
echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not running
    echo Please install Docker Desktop for Windows
    pause
    exit /b 1
)
echo [OK] Docker found
echo.

REM Check if Docker daemon is running
echo Checking Docker daemon...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running
    echo Please start Docker Desktop
    pause
    exit /b 1
)
echo [OK] Docker daemon is running
echo.

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
echo [OK] Directories ready
echo.

REM Stop any existing containers
echo Stopping existing containers if any...
docker-compose down 2>nul
echo.

REM Build Docker images
echo Building Docker images for 4 workers...
docker-compose build
if errorlevel 1 (
    echo [ERROR] Failed to build Docker images
    pause
    exit /b 1
)
echo [OK] Docker images built successfully
echo.

REM Start Docker containers
echo Starting 4 Docker containers...
echo   Worker 0 (Master): ml-worker-0
echo   Worker 1:         ml-worker-1
echo   Worker 2:         ml-worker-2
echo   Worker 3:         ml-worker-3
echo   Total: 4 containers (1 GPU per container)
echo.

docker-compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start containers
    pause
    exit /b 1
)

REM Wait for containers to be ready
echo Waiting for containers to initialize...
timeout /t 5 /nobreak >nul

REM Check container status
echo Container Status:
docker-compose ps
echo.

REM Show logs
echo ============================================================
echo Training Logs:
echo ============================================================
echo Press Ctrl+C to stop (containers will continue running)
echo.

docker-compose logs -f

