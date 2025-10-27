# Check for NVIDIA GPUs on Windows
# ==================================

Write-Host "============================================================"
Write-Host "Checking for NVIDIA GPUs"
Write-Host "============================================================"
Write-Host ""

# Check in Windows PowerShell
try {
    $gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpus) {
        Write-Host "Found NVIDIA GPU(s):" -ForegroundColor Green
        $gpus | ForEach-Object {
            Write-Host "  - $($_.Name)" -ForegroundColor Cyan
        }
        Write-Host ""
        Write-Host "Checking WSL2 GPU access..." -ForegroundColor Yellow
        $wslCheck = wsl nvidia-smi 2>&1
        if ($wslCheck -match "CUDA") {
            Write-Host "✓ GPU accessible in WSL2" -ForegroundColor Green
        } else {
            Write-Host "✗ GPU not accessible in WSL2" -ForegroundColor Red
            Write-Host "  You need to install NVIDIA CUDA drivers for WSL" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ No NVIDIA GPUs detected" -ForegroundColor Red
        Write-Host "  Training will run on CPU" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Could not check for GPUs" -ForegroundColor Red
}

Write-Host ""
Write-Host "For GPU support in Docker, you need:" -ForegroundColor Cyan
Write-Host "  1. NVIDIA GPU drivers installed"
Write-Host "  2. NVIDIA Container Toolkit installed in WSL2"
Write-Host "  3. Docker Desktop with WSL2 backend"
Write-Host ""

