#!/usr/bin/env bash
# train.sh — Pointer Network multi-stage training sequence
# Run from DL_MODEL/ptr_net/
#
# Usage:
#   ./train.sh                        # train all sizes (TSP only)
#   ./train.sh small                  # train one size (TSP only)
#   ./train.sh all --ood              # add Stage 3 (OOD generalisation, TSP)
#   ./train.sh all --tsptwd           # train all sizes for TSPTW-D
#   ./train.sh medium --ood --tsptwd  # OOD + TSPTW-D for one size
#
# Stage overview (TSP):
#   Stage 1 (n=8)           brute-force optimal labels, random instances
#   Stage 2 (n=10)          brute-force optimal labels, TSP dataset
#   Stage 3 (n=20,50, --ood) 2-opt NN labels, TSP dataset
#                            OOD generalisation: improves quality on unseen sizes
#
# Stage overview (TSPTW-D):
#   Stage T1 (n=8)   brute-force optimal labels on coords, random TSPTW-D features
#   Stage T2 (n=10)  optimal labels, random TSPTW-D features
#   Stage T3 (n=20)  nn2opt labels, TSPTW-D features
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
TSPTWD=0
for arg in "$@"; do
    [[ "$arg" == "--ood"    ]] && OOD=1
    [[ "$arg" == "--tsptwd" ]] && TSPTWD=1
done

run_stage() {
    echo ""
    echo "----------------------------------------------------"
    echo "  $1"
    echo "----------------------------------------------------"
    eval "$2"
}

train_tsp() {
    local s="$1"
    echo ""
    echo "===================================================="
    echo "  Training TSP: $s"
    echo "===================================================="

    # Stage 1: learn the pointer mechanism from brute-force optimal labels
    run_stage "[$s] Stage 1 — n=8, optimal labels, 1000 steps, random" \
        "$PYTHON train.py --mode tsp --size $s \
            --n 8 --label optimal --epochs 1000 --lr 1e-3 \
            --source random"

    # Stage 2: refine on real TSP instances, still within optimal-label range
    run_stage "[$s] Stage 2 — n=10, optimal labels, 2000 steps, tsp dataset" \
        "$PYTHON train.py --mode tsp --size $s --resume model/ptr_net_$s.pt \
            --n 10 --label optimal --epochs 2000 --lr 5e-4 \
            --source tsp"

    if [[ "$OOD" == "1" ]]; then
        echo ""
        echo "  OOD stages: model will attempt to generalise beyond training size."
        echo "  Quality may vary — use beam search decoding to get best results."

        # Stage 3a: small OOD with 2-opt labels
        run_stage "[$s] Stage 3a — n=20, nn2opt labels, 2000 steps, tsp dataset" \
            "$PYTHON train.py --mode tsp --size $s --resume model/ptr_net_$s.pt \
                --n 20 --label nn2opt --epochs 2000 --lr 2e-4 \
                --source tsp"

        # Stage 3b: medium OOD with 2-opt labels
        run_stage "[$s] Stage 3b — n=50, nn2opt labels, 2000 steps, tsp dataset" \
            "$PYTHON train.py --mode tsp --size $s --resume model/ptr_net_$s.pt \
                --n 50 --label nn2opt --epochs 2000 --lr 1e-4 \
                --source tsp"
    fi

    echo ""
    echo "  Done: model/ptr_net_$s.pt"
}

train_tsptwd() {
    local s="$1"
    echo ""
    echo "===================================================="
    echo "  Training TSPTW-D: $s"
    echo "===================================================="

    # Stage T1: learn from brute-force optimal labels on small TSPTW-D instances
    run_stage "[$s] Stage T1 — n=8, optimal labels, 1000 steps, tsptwd" \
        "$PYTHON train.py --mode tsptwd --size $s \
            --n 8 --label optimal --epochs 1000 --lr 1e-3"

    # Stage T2: refine on slightly larger instances
    run_stage "[$s] Stage T2 — n=10, optimal labels, 2000 steps, tsptwd" \
        "$PYTHON train.py --mode tsptwd --size $s --resume model/ptr_net_${s}_tsptwd.pt \
            --n 10 --label optimal --epochs 2000 --lr 5e-4"

    # Stage T3: push to larger instances with nn2opt labels
    run_stage "[$s] Stage T3 — n=20, nn2opt labels, 2000 steps, tsptwd" \
        "$PYTHON train.py --mode tsptwd --size $s --resume model/ptr_net_${s}_tsptwd.pt \
            --n 20 --label nn2opt --epochs 2000 --lr 2e-4"

    echo ""
    echo "  Done: model/ptr_net_${s}_tsptwd.pt"
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
print("      Use --tsptwd to also train TSPTW-D models.")
EOF

case "$SIZE" in
    all)
        if [[ "$TSPTWD" == "0" ]] || [[ "$TSPTWD" == "0" ]]; then
            train_tsp "small"; train_tsp "medium"; train_tsp "large"
        fi
        if [[ "$TSPTWD" == "1" ]]; then
            train_tsptwd "small"; train_tsptwd "medium"; train_tsptwd "large"
        fi
        ;;
    small|medium|large)
        train_tsp "$SIZE"
        [[ "$TSPTWD" == "1" ]] && train_tsptwd "$SIZE"
        ;;
    *) echo "Usage: $0 [small|medium|large|all] [--ood] [--tsptwd]" >&2; exit 1 ;;
esac

echo ""
echo "All training complete."
