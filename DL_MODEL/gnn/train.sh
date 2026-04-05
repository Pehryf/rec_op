#!/usr/bin/env bash
# train.sh -- GNN training sequence
# Run from DL_MODEL/gnn/
#
# Usage:
#   ./train.sh                 # train all sizes
#   ./train.sh small           # train one size
#   ./train.sh all --xl        # add Stage 4 (very large, needs GPU)
#
# Stage overview:
#   Stage 1 (n=8)           brute-force optimal labels, random instances
#   Stage 2 (n=10,50,100)   2-opt NN labels, TSP dataset
#   Stage 3 (n=150,300,500) plain NN labels, TSP dataset
#   Stage 4 (n=1000+, --xl) plain NN labels, TSP dataset (GPU only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

PYTHON=$(command -v python || command -v python3 || echo "")
[[ -z "$PYTHON" ]] && { echo "ERROR: python not found." >&2; exit 1; }

SIZE="${1:-all}"
XL=0
for arg in "$@"; do [[ "$arg" == "--xl" ]] && XL=1; done

run_stage() {
    echo ""
    echo "----------------------------------------------------"
    echo "  $1"
    echo "----------------------------------------------------"
    eval "$2"
}

train_model() {
    local s="$1"
    echo ""
    echo "===================================================="
    echo "  Training: $s"
    echo "===================================================="

    # Stage 1: learn basic tour structure from brute-force optimal labels
    run_stage "[$s] Stage 1 — n=8, optimal labels, 3000 steps" \
        "$PYTHON train.py --size $s \
            --n 8 --label optimal --steps 3000 --lr 1e-3 \
            --source random \
            --out model/gnn_$s.pt"

    # Stage 2: medium instances with 2-opt improved labels
    for n in 10 50 100; do
        run_stage "[$s] Stage 2 — n=$n, nn2opt labels, 3000 steps" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt \
                --n $n --label nn2opt --steps 3000 --lr 5e-4 \
                --source tsp \
                --out model/gnn_$s.pt"
    done

    # Stage 3: large instances with plain NN labels (2-opt too slow for n>300)
    for n in 150 300 500; do
        run_stage "[$s] Stage 3 — n=$n, nn labels, 3000 steps" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt \
                --n $n --label nn --steps 3000 --lr 1e-4 \
                --source tsp \
                --out model/gnn_$s.pt"
    done

    if [[ "$XL" == "1" ]]; then
        echo "  WARNING: Stage 4 needs significant VRAM (O(n²) edge tensor)."
        for n in 1000 3000 5000; do
            run_stage "[$s] Stage 4 — n=$n, nn labels, 300 steps" \
                "$PYTHON train.py --size $s --resume model/gnn_$s.pt \
                    --n $n --label nn --steps 300 --lr 5e-5 \
                    --source tsp \
                    --out model/gnn_$s.pt"
        done
    fi

    echo ""
    echo "  Done: model/gnn_$s.pt"
}

echo ""
echo "GNN Training Script"
$PYTHON - <<'EOF'
import torch
cuda = torch.cuda.is_available()
mps  = torch.backends.mps.is_available()
try:
    import intel_extension_for_pytorch; xpu = torch.xpu.is_available()
except ImportError:
    xpu = False
if cuda:   print(f"Device: cuda — {torch.cuda.get_device_name(0)}")
elif xpu:  print("Device: xpu  — Intel Arc (IPEX)")
elif mps:  print("Device: mps  — Apple Silicon")
else:      print("Device: cpu  — WARNING: no GPU, training will be slow.")
EOF

case "$SIZE" in
    all)           train_model "small"; train_model "medium"; train_model "large" ;;
    small|medium|large) train_model "$SIZE" ;;
    *) echo "Usage: $0 [small|medium|large|all] [--xl]" >&2; exit 1 ;;
esac

echo ""
echo "All training complete."
