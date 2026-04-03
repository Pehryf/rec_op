#!/usr/bin/env bash
# train.sh -- Full GNN training sequence
# Run from DL_MODEL/gnn/
#
# Usage:
#   ./train.sh                    # train all sizes (3 stages)
#   ./train.sh small              # train one size only
#   ./train.sh all --xl           # add Stage 4 -- very large instances (1000-5000 cities)
#   ./train.sh large --xl         # one size + Stage 4
#
# Each stage uses fixed n values so labels can be pre-generated and cached.
# Labels are stored per-n in model/labels_n<N>.npz and reused across runs.
#
# NOTE -- Stage 4 / --xl:
#   The GNN uses O(n^2) memory for edge embeddings.
#   n=1000 ~= 1 GB VRAM, n=5000 ~= 25 GB VRAM (large model).
#   Only run --xl on a GPU with sufficient VRAM.

set -euo pipefail

# Resolve python binary and activate venv if available
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

    # Stage 1: learn basic tour structure on small optimal instances (random source, brute-force labels)
    run_stage "[$s] Stage 1 - n=8, brute-force labels" \
        "$PYTHON train.py --size $s --n 8 --label optimal --steps 1000 --source random --label_cache model/labels_n8.npz --out model/gnn_$s.pt"

    # Stage 2: fixed-n medium instances from TSP dataset (labels pre-generated and cached per n)
    for n in 10 50 100; do
        run_stage "[$s] Stage 2 - n=$n, NN labels, TSP source" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt --n $n --label nn --steps 1000 --source tsp --lr 5e-4 --label_cache model/labels_n$n.npz --pool_cache model/city_pool.npy --out model/gnn_$s.pt"
    done

    # Stage 3: fixed-n large instances from TSP dataset
    for n in 150 300 500; do
        run_stage "[$s] Stage 3 - n=$n, NN labels, TSP source" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt --n $n --label nn --steps 1000 --source tsp --lr 1e-4 --label_cache model/labels_n$n.npz --pool_cache model/city_pool.npy --out model/gnn_$s.pt"
    done

    # Stage 4 (optional --xl): very large instances from TSP dataset
    if [[ "$XL" == "1" ]]; then
        echo ""
        echo "  WARNING: Stage 4 requires significant VRAM (n up to 5000, O(n^2) edges)."
        for n in 1000 3000 5000; do
            run_stage "[$s] Stage 4 - n=$n, NN labels, TSP source" \
                "$PYTHON train.py --size $s --resume model/gnn_$s.pt --n $n --label nn --steps 100 --source tsp --lr 5e-5 --label_cache model/labels_n$n.npz --pool_cache model/city_pool.npy --out model/gnn_$s.pt"
        done
    fi

    echo ""
    echo "  Done: model/gnn_$s.pt"
}

# Main
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
