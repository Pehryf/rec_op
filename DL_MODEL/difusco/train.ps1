# train.ps1 — DIFUSCO training sequence (TSPTW-D)
# Run from DL_MODEL/difusco/
#
# Usage:
#   .\train.ps1                    # train all sizes
#   .\train.ps1 -Size small        # train one size only
#   .\train.ps1 -TwoOpt            # improve labels with 2-opt (slower, better)
#   .\train.ps1 -XL                # add Stage 4 (very large n, GPU only)
#
# Stage overview:
#   Stage 1 (n=10)               pre-generated dataset instance, 2-opt labels
#   Stage 2 (n=20, 50, 100)     random instances, 2-opt / nn labels, 3000 steps each
#   Stage 3 (n=200, 300)        random instances, nn labels, 1000 steps each
#   Stage 4 (n=500, 1000, -XL)  random instances, nn labels, 300 steps (GPU only)
#
# NOTE — memory:
#   DIFUSCO uses O(n²) edge tensors (same as GNN) plus T diffusion steps.
#   n=300 is comfortably trainable on any modern GPU.
#   n=1000+ (-XL) requires significant VRAM.

param(
    [ValidateSet("small", "medium", "large", "all")]
    [string]$Size = "all",
    [switch]$XL,
    [switch]$TwoOpt
)

$ErrorActionPreference = "Stop"
$TwoOptFlag = if ($TwoOpt) { "--two_opt" } else { "" }

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
    $Out = "model/difusco_$S.pt"

    Write-Host ""
    Write-Host "====================================================" -ForegroundColor Green
    Write-Host "  Training DIFUSCO TSPTW-D: $S" -ForegroundColor Green
    Write-Host "====================================================" -ForegroundColor Green

    # Stage 1 — small pre-generated instance, warmup on structured data
    Run-Stage "[$S] Stage 1 - n=10, pre-generated dataset, 2-opt labels, 3000 steps" `
        "python train.py --size $S --n 10 --use_dataset --two_opt --epochs 3000 --lr 1e-3 --out $Out"

    # Stage 2 — medium random instances
    $LabelInfo = if ($TwoOpt) { "2-opt labels" } else { "nn labels" }
    foreach ($n in @(20, 50, 100)) {
        Run-Stage "[$S] Stage 2 - n=$n, random, $LabelInfo, 3000 steps" `
            "python train.py --size $S --resume $Out --n $n $TwoOptFlag --epochs 3000 --lr 5e-4 --out $Out"
    }

    # Stage 3 — larger random instances, plain NN labels
    foreach ($n in @(200, 300)) {
        Run-Stage "[$S] Stage 3 - n=$n, random, nn labels, 1000 steps" `
            "python train.py --size $S --resume $Out --n $n --epochs 1000 --lr 1e-4 --out $Out"
    }

    # Stage 4 (optional -XL) — very large instances, GPU only
    if ($XL) {
        Write-Host ""
        Write-Host "  WARNING: Stage 4 requires significant VRAM (O(n^2) edge tensor + diffusion)." -ForegroundColor Yellow
        foreach ($n in @(500, 1000)) {
            Run-Stage "[$S] Stage 4 - n=$n, random, nn labels, 300 steps" `
                "python train.py --size $S --resume $Out --n $n --epochs 300 --lr 5e-5 --out $Out"
        }
    }

    Write-Host ""
    Write-Host "  Done: $Out" -ForegroundColor Green
}

# ── Device info ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "DIFUSCO Training Script (TSPTW-D)" -ForegroundColor White
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
