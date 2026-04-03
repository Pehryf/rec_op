# train.ps1 — Full GNN training sequence
# Run from DL_MODEL/gnn/
#
# Usage:
#   .\train.ps1                    # train all sizes (3 stages)
#   .\train.ps1 -Size small        # train one size only
#   .\train.ps1 -XL                # add Stage 4 — very large instances (500-5000 cities)
#   .\train.ps1 -Size large -XL    # one size + Stage 4
#
# NOTE - Stage 4 / XL:
#   The GNN uses O(n^2) memory for edge embeddings.
#   n=1000 ~= 1 GB VRAM, n=5000 ~= 25 GB VRAM (large model).
#   Only run -XL on a GPU with sufficient VRAM, or reduce n_max accordingly.

param(
    [ValidateSet("small", "medium", "large", "all")]
    [string]$Size = "all",
    [switch]$XL
)

$ErrorActionPreference = "Stop"

function Run-Stage {
    param([string]$Label, [string]$Cmd)
    Write-Host ""
    Write-Host "----------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "----------------------------------------------------" -ForegroundColor Cyan
    Write-Host $Cmd -ForegroundColor Yellow
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Stage failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Train-Model {
    param([string]$S)

    Write-Host ""
    Write-Host "====================================================" -ForegroundColor Green
    Write-Host "  Training: $S" -ForegroundColor Green
    Write-Host "====================================================" -ForegroundColor Green

    # Stage 1: learn basic tour structure on small optimal instances
    $cmd = "python train.py --size $S --n 8 --label optimal --steps 1000 --source random --out model/gnn_$S.pt"
    Run-Stage "[$S] Stage 1 - small instances, brute-force labels" $cmd

    # Stage 2: expand to medium instances with NN labels
    $cmd = "python train.py --size $S --resume model/gnn_$S.pt --n_min 10 --n_max 100 --label nn --steps 3000 --source tsp --lr 5e-4 --out model/gnn_$S.pt"
    Run-Stage "[$S] Stage 2 - medium instances, NN labels, mixed sizes" $cmd

    # Stage 3: push to larger instances
    $cmd = "python train.py --size $S --resume model/gnn_$S.pt --n_min 50 --n_max 500 --label nn --steps 3000 --source tsp --lr 1e-4 --out model/gnn_$S.pt"
    Run-Stage "[$S] Stage 3 - large instances, NN labels, mixed sizes" $cmd

    # Stage 4 (optional -XL): very large instances
    if ($XL) {
        Write-Host ""
        Write-Host "  WARNING: Stage 4 requires significant VRAM (n up to 5000, O(n^2) edges)." -ForegroundColor Yellow
        $cmd = "python train.py --size $S --resume model/gnn_$S.pt --n_min 500 --n_max 5000 --label nn --steps 300 --source tsp --lr 5e-5 --out model/gnn_$S.pt"
        Run-Stage "[$S] Stage 4 - XL instances, NN labels, 500-5000 cities" $cmd
    }

    Write-Host ""
    Write-Host "  Done: model/gnn_$S.pt" -ForegroundColor Green
}

# Main
Write-Host ""
Write-Host "GNN Training Script" -ForegroundColor White
Write-Host "Checking device availability..." -ForegroundColor Gray

$deviceInfo = python -c @"
import torch
cuda = torch.cuda.is_available()
mps  = torch.backends.mps.is_available()
try:
    import intel_extension_for_pytorch
    xpu = torch.xpu.is_available()
except ImportError:
    xpu = False

if cuda:
    print(f'cuda  -- {torch.cuda.get_device_name(0)}')
elif xpu:
    print('xpu   -- Intel Arc (IPEX)')
elif mps:
    print('mps   -- Apple Silicon')
else:
    print('cpu   -- WARNING: no GPU detected. Training will be slow.')
    print('         NVIDIA : pip install torch --index-url https://download.pytorch.org/whl/cu124')
    print('         Intel  : pip install intel-extension-for-pytorch')
"@
Write-Host "Device: $deviceInfo" -ForegroundColor Cyan

if ($Size -eq "all") {
    Train-Model "small"
    Train-Model "medium"
    Train-Model "large"
} else {
    Train-Model $Size
}

Write-Host ""
Write-Host "All training complete." -ForegroundColor Green
