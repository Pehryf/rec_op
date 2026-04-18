# train.ps1 — Full GNN training sequence (TSP + TSPTW-D)
# Run from DL_MODEL/gnn/
#
# Usage:
#   .\train.ps1                         # train all sizes (TSP, 3 stages)
#   .\train.ps1 -Size small             # train one size only
#   .\train.ps1 -TSPTWD                 # also train TSPTW-D variants
#   .\train.ps1 -TSPTWD -FineTune       # fine-tune existing TSPTW-D model on large n only
#   .\train.ps1 -XL                     # add Stage 4 (very large instances, GPU only)
#
# Stage overview:
#   Stage 1 (n=8)            optimal labels, random instances
#   Stage 2 (n=10,50,100)    nn2opt labels, TSP dataset
#   Stage 3 (n=150,300,500)  nn labels, TSP dataset
#   Stage 4 (n=1000+, -XL)   nn labels, TSP dataset (GPU only)
#   TSPTW-D stages           same progression with --mode tsptwd (node_dim=5, edge_dim=4)
#   FineTune                 skip small-n stages, mixed n_min=50..500, lr=1e-4
#
# NOTE - Stage 4 / XL:
#   The GNN uses O(n^2) memory for edge embeddings.
#   n=1000 ~= 1 GB VRAM, n=5000 ~= 25 GB VRAM (large model).
#   Only run -XL on a GPU with sufficient VRAM.

param(
    [ValidateSet("small", "medium", "large", "all")]
    [string]$Size = "all",
    [switch]$XL,
    [switch]$TSPTWD,
    [switch]$FineTune
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
    Write-Host "  Training TSP: $S" -ForegroundColor Green
    Write-Host "====================================================" -ForegroundColor Green

    $TspModel = "model/gnn_$S.pt"

    if (Test-Path $TspModel) {
        Write-Host ""
        Write-Host "  Existing TSP model found: $TspModel" -ForegroundColor Yellow
        Write-Host "  Skipping TSP stages." -ForegroundColor Yellow
    } else {
        # Stage 1: basic tour structure on small optimal instances
        Run-Stage "[$S] Stage 1 - n=8, brute-force labels" `
            "python train.py --size $S --mode tsp --n 8 --label optimal --steps 3000 --source random --out $TspModel"

        # Stage 2: medium instances (2-opt improved labels)
        foreach ($n in @(10, 50, 100)) {
            Run-Stage "[$S] Stage 2 - n=$n, nn2opt labels, TSP source" `
                "python train.py --size $S --mode tsp --resume $TspModel --n $n --label nn2opt --steps 3000 --lr 5e-4 --source tsp --out $TspModel"
        }

        # Stage 3: large instances (plain NN labels)
        foreach ($n in @(150, 300, 500)) {
            Run-Stage "[$S] Stage 3 - n=$n, nn labels, TSP source" `
                "python train.py --size $S --mode tsp --resume $TspModel --n $n --label nn --steps 3000 --lr 1e-4 --source tsp --out $TspModel"
        }

        # Stage 4 (optional -XL): very large instances
        if ($XL) {
            Write-Host ""
            Write-Host "  WARNING: Stage 4 requires significant VRAM (O(n^2) edge tensor)." -ForegroundColor Yellow
            foreach ($n in @(1000, 3000, 5000)) {
                Run-Stage "[$S] Stage 4 - n=$n, nn labels, TSP source" `
                    "python train.py --size $S --mode tsp --resume $TspModel --n $n --label nn --steps 300 --lr 5e-5 --source tsp --out $TspModel"
            }
        }

        Write-Host ""
        Write-Host "  Done: $TspModel" -ForegroundColor Green
    }

    # TSPTW-D variant (--mode tsptwd, node_dim=5, edge_dim=4)
    # Trained separately from the TSP model - never overwrites gnn_$S.pt.
    # If gnn_${S}_tsptwd.pt already exists, Stage 1 is skipped (fine-tune).
    # Use -FineTune to skip small-n stages and go straight to large-n mixed training.
    if ($TSPTWD) {
        Write-Host ""
        Write-Host "====================================================" -ForegroundColor Magenta
        Write-Host "  Training TSPTW-D: $S" -ForegroundColor Magenta
        Write-Host "====================================================" -ForegroundColor Magenta

        $TsptWdModel = "model/gnn_${S}_tsptwd.pt"

        if ($FineTune) {
            if (-not (Test-Path $TsptWdModel)) {
                Write-Host "  ERROR: -FineTune requires an existing model at $TsptWdModel" -ForegroundColor Red
                exit 1
            }
            Write-Host ""
            Write-Host "  Fine-tune mode: skipping small-n stages." -ForegroundColor Yellow
            Write-Host "  Resuming from: $TsptWdModel" -ForegroundColor Yellow

            Run-Stage "[$S] TSPTWD FineTune - n_min=50 n_max=500, nn labels, JSON source, 5000 steps" `
                "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n_min 50 --n_max 500 --label nn --steps 5000 --lr 1e-4 --source tsptwd_json --out $TsptWdModel"

            Run-Stage "[$S] TSPTWD FineTune - n_min=300 n_max=1000, nn labels, JSON source, 3000 steps" `
                "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n_min 300 --n_max 1000 --label nn --steps 3000 --lr 5e-5 --source tsptwd_json --out $TsptWdModel"
        } else {
            if (Test-Path $TsptWdModel) {
                Write-Host ""
                Write-Host "  Existing model found: $TsptWdModel" -ForegroundColor Yellow
                Write-Host "  Skipping Stage 1 - fine-tuning from existing weights." -ForegroundColor Yellow
            } else {
                Run-Stage "[$S] TSPTWD Stage 1 - n=8, optimal labels (fresh start)" `
                    "python train.py --size $S --mode tsptwd --n 8 --label optimal --steps 3000 --source random --out $TsptWdModel"
            }

            foreach ($n in @(10, 50, 100)) {
                Run-Stage "[$S] TSPTWD Stage 2 - n=$n, nn2opt labels, JSON source" `
                    "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n $n --label nn2opt --steps 3000 --lr 5e-4 --source tsptwd_json --out $TsptWdModel"
            }

            foreach ($n in @(150, 300, 500)) {
                Run-Stage "[$S] TSPTWD Stage 3 - n=$n, nn labels, JSON source" `
                    "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n $n --label nn --steps 3000 --lr 1e-4 --source tsptwd_json --out $TsptWdModel"
            }

            Run-Stage "[$S] TSPTWD Stage 3 - n=700, nn labels, JSON source" `
                "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n 700 --label nn --steps 2000 --lr 5e-5 --source tsptwd_json --out $TsptWdModel"

            Run-Stage "[$S] TSPTWD Stage 3 - n=1000, nn labels, JSON source" `
                "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n 1000 --label nn --steps 1500 --lr 5e-5 --source tsptwd_json --out $TsptWdModel"

            if ($XL) {
                Write-Host ""
                Write-Host "  WARNING: Stage 4 requires significant VRAM (O(n^2) edge tensor)." -ForegroundColor Yellow
                foreach ($n in @(1000, 3000, 5000)) {
                    Run-Stage "[$S] TSPTWD Stage 4 - n=$n, nn labels, JSON source" `
                        "python train.py --size $S --mode tsptwd --resume $TsptWdModel --n $n --label nn --steps 300 --lr 5e-5 --source tsptwd_json --out $TsptWdModel"
                }
            }
        }

        Write-Host ""
        Write-Host "  Done: $TsptWdModel" -ForegroundColor Magenta
    }
}

# Main
Write-Host ""
Write-Host "GNN Training Script (TSP + TSPTW-D)" -ForegroundColor White
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
