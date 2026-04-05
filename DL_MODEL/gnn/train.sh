#!/usr/bin/env bash
# train.sh -- Full GNN training sequence
# Run from DL_MODEL/gnn/
#
# Usage:
#   ./train.sh                    # train all sizes (stages 1–3)
#   ./train.sh small              # train one size only
#   ./train.sh all --xl           # add Stage 4 — very large instances (1000+ cities)
#   ./train.sh large --xl         # one size + Stage 4
#
# Label caches are stored per-n in model/labels_n<N>.npz.
# If a cache exists it is reused, so delete it to rebuild with different labels:
#   rm model/labels_n50.npz  # will be rebuilt with nn2opt on next run
#
# Stage overview:
#   Stage 1 (n=8)          : brute-force optimal labels — learn basic tour structure
#   Stage 2 (n=10,50,100)  : 2-opt improved NN labels  — learn medium-scale tours
#   Stage 3 (n=150,300,500): plain NN labels            — scale up to large instances
#   Stage 4 (n=1000+, --xl): plain NN labels            — very large (needs ≥8 GB VRAM)
#
# NOTE on Stage 4 / --xl:
#   The GNN uses O(n²) memory for edge embeddings.
#   n=1000 ≈ 1 GB VRAM; only run on a GPU with sufficient memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv"
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi

PYTHON=$(command -v python || command -v python3 || echo "")
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: python not found. Activate your venv or install Python." >&2
    exit 1
fi

SIZE="${1:-all}"
XL=0
for arg in "$@"; do
    [[ "$arg" == "--xl" ]] && XL=1
done

run_stage() {
    local label="$1"
    local cmd="$2"
    echo ""
    echo "----------------------------------------------------"
    echo "  $label"
    echo "----------------------------------------------------"
    echo "$cmd"
    eval "$cmd"
}

train_model() {
    local s="$1"

    echo ""
    echo "===================================================="
    echo "  Training: $s"
    echo "===================================================="

    # Stage 1 — brute-force optimal labels on small instances.
    # 2000 steps gives the model a solid foundation.
    run_stage "[$s] Stage 1 — n=8, optimal labels" \
        "$PYTHON train.py --size $s \
            --n 8 --label optimal --steps 2000 \
            --source random \
            --label_cache model/labels_n8.npz \
            --out model/gnn_$s.pt"

    # Stage 2 — 2-opt improved NN labels for medium instances.
    # nn2opt produces ~10-20% shorter tours than plain NN: much better teacher.
    # 2000 steps per size; lr reduced to 5e-4 to avoid forgetting Stage 1.
    for n in 10 50 100; do
        run_stage "[$s] Stage 2 — n=$n, nn2opt labels, TSP source" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt \
                --n $n --label nn2opt --steps 2000 \
                --source tsp --lr 5e-4 \
                --label_cache model/labels_n${n}_2opt.npz \
                --pool_cache model/city_pool.npy \
                --out model/gnn_$s.pt"
    done

    # Stage 3 — plain NN labels for large instances (2-opt is too slow for n>300).
    # lr further reduced to 1e-4 for fine-scale adaptation.
    for n in 150 300 500; do
        run_stage "[$s] Stage 3 — n=$n, nn labels, TSP source" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt \
                --n $n --label nn --steps 2000 \
                --source tsp --lr 1e-4 \
                --label_cache model/labels_n$n.npz \
                --pool_cache model/city_pool.npy \
                --out model/gnn_$s.pt"
    done

    # Stage 4 (optional --xl) — very large instances
    if [[ "$XL" == "1" ]]; then
        echo ""
        echo "  WARNING: Stage 4 requires significant VRAM (n up to 5000, O(n²) edges)."
        for n in 1000 3000 5000; do
            run_stage "[$s] Stage 4 — n=$n, nn labels, TSP source" \
                "$PYTHON train.py --size $s --resume model/gnn_$s.pt \
                    --n $n --label nn --steps 200 \
                    --source tsp --lr 5e-5 \
                    --label_cache model/labels_n$n.npz \
                    --pool_cache model/city_pool.npy \
                    --out model/gnn_$s.pt"
        done
    fi

    echo ""
    echo "  Done: model/gnn_$s.pt"
}

# ── Device check ──────────────────────────────────────────────────────────────
echo ""
echo "GNN Training Script"
echo "Checking device availability..."

$PYTHON - <<'EOF'
import torch
cuda = torch.cuda.is_available()
mps  = torch.backends.mps.is_available()
try:
    import intel_extension_for_pytorch
    xpu = torch.xpu.is_available()
except ImportError:
    xpu = False

if cuda:
    print(f"Device: cuda  -- {torch.cuda.get_device_name(0)}")
elif xpu:
    print("Device: xpu   -- Intel Arc (IPEX)")
elif mps:
    print("Device: mps   -- Apple Silicon")
else:
    print("Device: cpu   -- WARNING: no GPU detected. Training will be slow.")
    print("               NVIDIA : pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("               Intel  : pip install intel-extension-for-pytorch")
EOF

# ── Run ───────────────────────────────────────────────────────────────────────
case "$SIZE" in
    all)
        train_model "small"
        train_model "medium"
        train_model "large"
        ;;
    small|medium|large)
        train_model "$SIZE"
        ;;
    *)
        echo "Usage: $0 [small|medium|large|all] [--xl]" >&2
        exit 1
        ;;
esac

echo ""
echo "All training complete."
