# train.ps1 — Full GNN training sequence
# Run from DL_MODEL/gnn/
#
# Usage:
#   .\train.ps1                    # train all sizes (3 stages)
#   .\train.ps1 -Size small        # train one size only
#   .\train.ps1 -XL                # add Stage 4 — very large instances (1000-5000 cities)
#   .\train.ps1 -Size large -XL    # one size + Stage 4
#
# Each stage uses fixed n values so labels can be pre-generated and cached.
# Labels are stored per-n in model/labels_n<N>.npz and reused across runs.
#
# NOTE - Stage 4 / XL:
#   The GNN uses O(n^2) memory for edge embeddings.
#   n=1000 ~= 1 GB VRAM, n=5000 ~= 25 GB VRAM (large model).
#   Only run -XL on a GPU with sufficient VRAM.

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

    # Stage 1: learn basic tour structure on small optimal instances (random source, brute-force labels)
    $cmd = "python train.py --size $S --n 8 --label optimal --steps 1000 --source random --label_cache model/labels_n8.npz --out model/gnn_$S.pt"
    Run-Stage "[$S] Stage 1 - n=8, brute-force labels" $cmd

    # Stage 2: fixed-n medium instances from TSP dataset (labels pre-generated and cached per n)
    foreach ($n in @(10, 50, 100)) {
        $cmd = "python train.py --size $S --resume model/gnn_$S.pt --n $n --label nn --steps 1000 --source tsp --lr 5e-4 --label_cache model/labels_n$n.npz --pool_cache model/city_pool.npy --out model/gnn_$S.pt"
        Run-Stage "[$S] Stage 2 - n=$n, NN labels, TSP source" $cmd
    }

    # Stage 3: fixed-n large instances from TSP dataset
    foreach ($n in @(150, 300, 500)) {
        $cmd = "python train.py --size $S --resume model/gnn_$S.pt --n $n --label nn --steps 1000 --source tsp --lr 1e-4 --label_cache model/labels_n$n.npz --pool_cache model/city_pool.npy --out model/gnn_$S.pt"
        Run-Stage "[$S] Stage 3 - n=$n, NN labels, TSP source" $cmd
    }

    # Stage 4 (optional -XL): very large instances from TSP dataset
    if ($XL) {
        Write-Host ""
        Write-Host "  WARNING: Stage 4 requires significant VRAM (n up to 5000, O(n^2) edges)." -ForegroundColor Yellow
        foreach ($n in @(1000, 3000, 5000)) {
            $cmd = "python train.py --size $S --resume model/gnn_$S.pt --n $n --label nn --steps 100 --source tsp --lr 5e-5 --label_cache model/labels_n$n.npz --pool_cache model/city_pool.npy --out model/gnn_$S.pt"
            Run-Stage "[$S] Stage 4 - n=$n, NN labels, TSP source" $cmd
        }
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
