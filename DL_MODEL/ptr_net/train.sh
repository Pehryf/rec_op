#!/usr/bin/env bash
# train.sh — Pointer Network multi-stage training sequence
# Run from DL_MODEL/ptr_net/
#
# Usage:
#   ./train.sh                 # train all sizes
#   ./train.sh small           # train one size
#   ./train.sh all --ood       # add Stage 3 (OOD generalisation attempt)
#
# Stage overview:
#   Stage 1 (n=8)           brute-force optimal labels, random instances
#   Stage 2 (n=10)          brute-force optimal labels, TSP dataset
#   Stage 3 (n=20,50, --ood) 2-opt NN labels, TSP dataset
#                            OOD generalisation: improves quality on unseen sizes
#                            at a small cost in training time
#
# Note on label constraints
# -------------------------
# Optimal labels require brute-force (O(n!)) — only feasible for n ≤ 10.
# For n > 10, use --label nn2opt (NN + 2-opt pseudo-labels).
# Unlike the GNN (which uses edge-level BCE), Ptr-Net is trained with
# cross-entropy over decoding steps, so label quality matters less at larger n
# — the model learns the pointer mechanism, not exact tour structure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

PYTHON=$(command -v python || command -v python3 || echo "")
[[ -z "$PYTHON" ]] && { echo "ERROR: python not found." >&2; exit 1; }

SIZE="${1:-all}"
OOD=0
for arg in "$@"; do [[ "$arg" == "--ood" ]] && OOD=1; done

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

    # Stage 1: learn the pointer mechanism from brute-force optimal labels
    run_stage "[$s] Stage 1 — n=8, optimal labels, 1000 steps, random" \
        "$PYTHON train.py --size $s \
            --n 8 --label optimal --steps 1000 --lr 1e-3 \
            --source random \
            --out model/ptr_net_$s.pt"

    # Stage 2: refine on real TSP instances, still within optimal-label range
    run_stage "[$s] Stage 2 — n=10, optimal labels, 2000 steps, tsp dataset" \
        "$PYTHON train.py --size $s --resume model/ptr_net_$s.pt \
            --n 10 --label optimal --steps 2000 --lr 5e-4 \
            --source tsp \
            --out model/ptr_net_$s.pt"

    if [[ "$OOD" == "1" ]]; then
        echo ""
        echo "  OOD stages: model will attempt to generalise beyond training size."
        echo "  Quality may vary — use beam search decoding to get best results."

        # Stage 3a: small OOD with 2-opt labels
        run_stage "[$s] Stage 3a — n=20, nn2opt labels, 2000 steps, tsp dataset" \
            "$PYTHON train.py --size $s --resume model/ptr_net_$s.pt \
                --n 20 --label nn2opt --steps 2000 --lr 2e-4 \
                --source tsp \
                --out model/ptr_net_$s.pt"

        # Stage 3b: medium OOD with 2-opt labels
        run_stage "[$s] Stage 3b — n=50, nn2opt labels, 2000 steps, tsp dataset" \
            "$PYTHON train.py --size $s --resume model/ptr_net_$s.pt \
                --n 50 --label nn2opt --steps 2000 --lr 1e-4 \
                --source tsp \
                --out model/ptr_net_$s.pt"
    fi

    echo ""
    echo "  Done: model/ptr_net_$s.pt"
}

echo ""
echo "Pointer Network Training Script"
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
print()
print("Note: Ptr-Net labels require n ≤ 10 for optimal (brute-force).")
print("      Use --ood to add Stage 3 (nn2opt labels, n=20/50).")
EOF

case "$SIZE" in
    all)                    train_model "small"; train_model "medium"; train_model "large" ;;
    small|medium|large)     train_model "$SIZE" ;;
    *) echo "Usage: $0 [small|medium|large|all] [--ood]" >&2; exit 1 ;;
esac

echo ""
echo "All training complete."
