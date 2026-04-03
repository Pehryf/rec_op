#!/usr/bin/env bash
# train.sh -- Full GNN training sequence
# Run from DL_MODEL/gnn/
#
# Usage:
#   ./train.sh                    # train all sizes (3 stages)
#   ./train.sh small              # train one size only
#   ./train.sh all --xl           # add Stage 4 -- very large instances (500-5000 cities)
#   ./train.sh large --xl         # one size + Stage 4
#
# NOTE -- Stage 4 / --xl:
#   The GNN uses O(n^2) memory for edge embeddings.
#   n=1000 ~= 1 GB VRAM, n=5000 ~= 25 GB VRAM (large model).
#   Only run --xl on a GPU with sufficient VRAM, or reduce n_max accordingly.

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

    # Stage 1: learn basic tour structure on small optimal instances
    run_stage "[$s] Stage 1 - small instances, brute-force labels" \
        "$PYTHON train.py --size $s --n 8 --label optimal --steps 1000 --source random --label_cache model/labels_n8.npz --out model/gnn_$s.pt"

    # Stage 2: expand to medium instances with NN labels
    run_stage "[$s] Stage 2 - medium instances, NN labels, mixed sizes" \
        "$PYTHON train.py --size $s --resume model/gnn_$s.pt --n_min 10 --n_max 100 --label nn --steps 3000 --source tsp --lr 5e-4 --pool_cache model/city_pool.npy --out model/gnn_$s.pt"

    # Stage 3: push to larger instances
    run_stage "[$s] Stage 3 - large instances, NN labels, mixed sizes" \
        "$PYTHON train.py --size $s --resume model/gnn_$s.pt --n_min 50 --n_max 500 --label nn --steps 3000 --source tsp --lr 1e-4 --pool_cache model/city_pool.npy --out model/gnn_$s.pt"

    # Stage 4 (optional --xl): very large instances
    if [[ "$XL" == "1" ]]; then
        echo ""
        echo "  WARNING: Stage 4 requires significant VRAM (n up to 5000, O(n^2) edges)."
        run_stage "[$s] Stage 4 - XL instances, NN labels, 500-5000 cities" \
            "$PYTHON train.py --size $s --resume model/gnn_$s.pt --n_min 500 --n_max 5000 --label nn --steps 300 --source tsp --lr 5e-5 --pool_cache model/city_pool.npy --out model/gnn_$s.pt"
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
