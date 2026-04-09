#!/usr/bin/env bash
# train.sh — DIFUSCO training sequence (TSPTW-D)
# Run from DL_MODEL/difusco/
#
# Usage:
#   ./train.sh                      # train all sizes
#   ./train.sh small                # train one size only
#   ./train.sh all --two_opt        # improve labels with 2-opt (slower, better)
#   ./train.sh all --xl             # add Stage 4 (very large n, needs GPU)
#
# Stage overview:
#   Stage 1 (n=10)              pre-generated dataset instance, 2-opt labels
#   Stage 2 (n=20, 50, 100)    random instances, 2-opt labels, 3000 steps each
#   Stage 3 (n=200, 300)       random instances, nn labels,   1000 steps each
#   Stage 4 (n=500, 1000, --xl) random instances, nn labels,   300 steps (GPU only)
#
# NOTE — memory:
#   DIFUSCO uses O(n²) edge tensors (same as GNN) plus T diffusion steps.
#   n=300 is comfortably trainable on any modern GPU.
#   n=1000+ (--xl) requires significant VRAM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

PYTHON=$(command -v python || command -v python3 || echo "")
[[ -z "$PYTHON" ]] && { echo "ERROR: python not found." >&2; exit 1; }

SIZE="${1:-all}"
XL=0; TWO_OPT=0
for arg in "$@"; do
    [[ "$arg" == "--xl" ]]      && XL=1
    [[ "$arg" == "--two_opt" ]] && TWO_OPT=1
done

TWO_OPT_FLAG=$( [[ "$TWO_OPT" == "1" ]] && echo "--two_opt" || echo "" )

run_stage() {
    echo ""
    echo "----------------------------------------------------"
    echo "  $1"
    echo "----------------------------------------------------"
    eval "$2"
}

train_model() {
    local s="$1"
    local out="model/difusco_$s.pt"

    echo ""
    echo "===================================================="
    echo "  Training DIFUSCO TSPTW-D: $s"
    echo "===================================================="

    # Stage 1 — small pre-generated instance, warmup on structured data
    run_stage "[$s] Stage 1 — n=10, pre-generated dataset, 2-opt labels, 3000 steps" \
        "$PYTHON train.py --size $s \
            --n 10 --use_dataset --two_opt --epochs 3000 --lr 1e-3 \
            --out $out"

    # Stage 2 — medium random instances, 2-opt labels
    for n in 20 50 100; do
        run_stage "[$s] Stage 2 — n=$n, random, ${TWO_OPT_FLAG:-nn} labels, 3000 steps" \
            "$PYTHON train.py --size $s --resume $out \
                --n $n $TWO_OPT_FLAG --epochs 3000 --lr 5e-4 \
                --out $out"
    done

    # Stage 3 — larger random instances, plain NN labels
    for n in 200 300; do
        run_stage "[$s] Stage 3 — n=$n, random, nn labels, 1000 steps" \
            "$PYTHON train.py --size $s --resume $out \
                --n $n --epochs 1000 --lr 1e-4 \
                --out $out"
    done

    # Stage 4 (optional --xl) — very large instances, GPU only
    if [[ "$XL" == "1" ]]; then
        echo "  WARNING: Stage 4 needs significant VRAM (O(n²) edge tensor + diffusion)."
        for n in 500 1000; do
            run_stage "[$s] Stage 4 — n=$n, random, nn labels, 300 steps" \
                "$PYTHON train.py --size $s --resume $out \
                    --n $n --epochs 300 --lr 5e-5 \
                    --out $out"
        done
    fi

    echo ""
    echo "  Done: $out"
}

# ── Device info ───────────────────────────────────────────────────────────────
echo ""
echo "DIFUSCO Training Script (TSPTW-D)"
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
    all)                    train_model "small"; train_model "medium"; train_model "large" ;;
    small|medium|large)     train_model "$SIZE" ;;
    *) echo "Usage: $0 [small|medium|large|all] [--two_opt] [--xl]" >&2; exit 1 ;;
esac

echo ""
echo "All training complete."
