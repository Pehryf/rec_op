# train.ps1 — Pointer Network multi-stage training sequence (Windows PowerShell)
# Run from DL_MODEL/ptr_net/
#
# Usage:
#   .\train.ps1                          # train all sizes (TSP only)
#   .\train.ps1 -Size small              # train one size (TSP only)
#   .\train.ps1 -Size all -Ood           # add Stage 3 (OOD generalisation, TSP)
#   .\train.ps1 -Size all -Tsptwd        # train all sizes for TSPTW-D
#   .\train.ps1 -Size medium -Ood -Tsptwd  # OOD + TSPTW-D for one size

param(
    [string]$Size   = "all",
    [switch]$Ood,
    [switch]$Tsptwd
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv      = Join-Path $ScriptDir "..\..\\.venv"
if (Test-Path "$Venv\Scripts\Activate.ps1") {
    & "$Venv\Scripts\Activate.ps1"
}

$Python = (Get-Command python -ErrorAction SilentlyContinue)?.Source `
        ?? (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
if (-not $Python) { Write-Error "ERROR: python not found."; exit 1 }

function Run-Stage($Label, $Cmd) {
    Write-Host ""
    Write-Host "----------------------------------------------------"
    Write-Host "  $Label"
    Write-Host "----------------------------------------------------"
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Label" }
}

function Train-TSP($S) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "  Training TSP: $S"
    Write-Host "===================================================="

    Run-Stage "[$S] Stage 1 — n=8, optimal labels, 1000 steps, random" `
        "$Python train.py --mode tsp --size $S --n 8 --label optimal --epochs 1000 --lr 1e-3 --source random"

    Run-Stage "[$S] Stage 2 — n=10, optimal labels, 2000 steps, tsp dataset" `
        "$Python train.py --mode tsp --size $S --resume model/ptr_net_$S.pt --n 10 --label optimal --epochs 2000 --lr 5e-4 --source tsp"

    if ($Ood) {
        Write-Host ""
        Write-Host "  OOD stages: model will attempt to generalise beyond training size."

        Run-Stage "[$S] Stage 3a — n=20, nn2opt labels, 2000 steps, tsp dataset" `
            "$Python train.py --mode tsp --size $S --resume model/ptr_net_$S.pt --n 20 --label nn2opt --epochs 2000 --lr 2e-4 --source tsp"

        Run-Stage "[$S] Stage 3b — n=50, nn2opt labels, 2000 steps, tsp dataset" `
            "$Python train.py --mode tsp --size $S --resume model/ptr_net_$S.pt --n 50 --label nn2opt --epochs 2000 --lr 1e-4 --source tsp"
    }

    Write-Host ""
    Write-Host "  Done: model/ptr_net_$S.pt"
}

function Train-TSPTWD($S) {
    Write-Host ""
    Write-Host "===================================================="
    Write-Host "  Training TSPTW-D: $S"
    Write-Host "===================================================="

    Run-Stage "[$S] Stage T1 — n=8, optimal labels, 1000 steps, tsptwd" `
        "$Python train.py --mode tsptwd --size $S --n 8 --label optimal --epochs 1000 --lr 1e-3"

    Run-Stage "[$S] Stage T2 — n=10, optimal labels, 2000 steps, tsptwd" `
        "$Python train.py --mode tsptwd --size $S --resume model/ptr_net_${S}_tsptwd.pt --n 10 --label optimal --epochs 2000 --lr 5e-4"

    Run-Stage "[$S] Stage T3 — n=20, nn2opt labels, 2000 steps, tsptwd" `
        "$Python train.py --mode tsptwd --size $S --resume model/ptr_net_${S}_tsptwd.pt --n 20 --label nn2opt --epochs 2000 --lr 2e-4"

    Write-Host ""
    Write-Host "  Done: model/ptr_net_${S}_tsptwd.pt"
}

# Device info
& $Python -c @"
import torch
cuda = torch.cuda.is_available()
mps  = torch.backends.mps.is_available()
try:
    import intel_extension_for_pytorch; xpu = torch.xpu.is_available()
except ImportError:
    xpu = False
if cuda:   print(f'Device: cuda -- {torch.cuda.get_device_name(0)}')
elif xpu:  print('Device: xpu  -- Intel Arc (IPEX)')
elif mps:  print('Device: mps  -- Apple Silicon')
else:      print('Device: cpu  -- WARNING: no GPU, training will be slow.')
print()
print('Note: Ptr-Net labels require n <= 10 for optimal (brute-force).')
print('      Use -Ood to add Stage 3 (nn2opt labels, n=20/50).')
print('      Use -Tsptwd to also train TSPTW-D models.')
"@

$Sizes = if ($Size -eq "all") { @("small", "medium", "large") } else { @($Size) }
foreach ($s in $Sizes) {
    if ($s -notin @("small","medium","large")) {
        Write-Error "Unknown size '$s'. Use small, medium, large, or all."; exit 1
    }
    Train-TSP $s
    if ($Tsptwd) { Train-TSPTWD $s }
}

Write-Host ""
Write-Host "All training complete."
