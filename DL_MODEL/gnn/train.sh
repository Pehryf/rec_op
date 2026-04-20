#!/usr/bin/env bash
# train.sh — GNN training sequence (TSP + TSPTW-D)
# Run from DL_MODEL/gnn/
#
# Usage:
#   ./train.sh                          # train all sizes (TSP)
#   ./train.sh small                    # train one size
#   ./train.sh all --tsptwd             # also train TSPTW-D variants
#   ./train.sh all --tsptwd --finetune  # fine-tune existing TSPTW-D on large n only
#   ./train.sh all --xl                 # add Stage 4 (very large, needs GPU)
#
# Stage overview:
#   Stage 1 (n=8)            optimal labels, random instances
#   Stage 2 (n=10,50,100)    nn2opt labels, TSP dataset
#   Stage 3 (n=150,300,500)  nn labels, TSP dataset
#   Stage 4 (n=1000+, --xl)  nn labels, TSP dataset (GPU only)
#   TSPTW-D stages           same progression with --mode tsptwd (node_dim=5, edge_dim=4)
#   --finetune               skip small-n stages, mixed n_min=50..500, lr=1e-4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

PYTHON=$(command -v python || command -v python3 || echo "")
[[ -z "$PYTHON" ]] && { echo "ERROR: python not found." >&2; exit 1; }

SIZE="${1:-all}"
XL=0; TSPTWD=0; FINETUNE=0
for arg in "$@"; do
    [[ "$arg" == "--xl" ]]        && XL=1
    [[ "$arg" == "--tsptwd" ]]    && TSPTWD=1
    [[ "$arg" == "--finetune" ]]  && FINETUNE=1
done

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
    echo "  Training TSP: $s"
    echo "===================================================="

    local tsp_model="model/gnn_$s.pt"

    if [[ -f "$tsp_model" ]]; then
        echo ""
        echo "  Existing TSP model found: $tsp_model"
        echo "  Skipping TSP stages."
    else
        run_stage "[$s] Stage 1 - n=8, optimal labels, 3000 steps" \
            "$PYTHON train.py --size $s --mode tsp \
                --n 8 --label optimal --steps 3000 --lr 1e-3 \
                --source random \
                --out $tsp_model"

        for n in 10 50 100; do
            run_stage "[$s] Stage 2 - n=$n, nn2opt labels, 3000 steps" \
                "$PYTHON train.py --size $s --mode tsp --resume $tsp_model \
                    --n $n --label nn2opt --steps 3000 --lr 5e-4 \
                    --source tsp \
                    --out $tsp_model"
        done

        for n in 150 300 500; do
            run_stage "[$s] Stage 3 - n=$n, nn labels, 3000 steps" \
                "$PYTHON train.py --size $s --mode tsp --resume $tsp_model \
                    --n $n --label nn --steps 3000 --lr 1e-4 \
                    --source tsp \
                    --out $tsp_model"
        done

        if [[ "$XL" == "1" ]]; then
            echo "  WARNING: Stage 4 needs significant VRAM (O(n^2) edge tensor)."
            for n in 1000 3000 5000; do
                run_stage "[$s] Stage 4 - n=$n, nn labels, 300 steps" \
                    "$PYTHON train.py --size $s --mode tsp --resume $tsp_model \
                        --n $n --label nn --steps 300 --lr 5e-5 \
                        --source tsp \
                        --out $tsp_model"
            done
        fi

        echo ""
        echo "  Done: $tsp_model"
    fi

    # TSPTW-D variant (--mode tsptwd, node_dim=5, edge_dim=4)
    # Trained separately from the TSP model — never overwrites gnn_$s.pt.
    # If gnn_${s}_tsptwd.pt already exists, Stage 1 is skipped (fine-tune).
    # Use --finetune to skip small-n stages and go straight to large-n mixed training.
    if [[ "$TSPTWD" == "1" ]]; then
        echo ""
        echo "===================================================="
        echo "  Training TSPTW-D: $s"
        echo "===================================================="

        local tsptwd_model="model/gnn_${s}_tsptwd.pt"

        local SEED
        SEED=$(date +%s)

        if [[ "$FINETUNE" == "1" ]]; then
            if [[ ! -f "$tsptwd_model" ]]; then
                echo "  ERROR: --finetune requires an existing model at $tsptwd_model" >&2
                exit 1
            fi
            echo ""
            echo "  Fine-tune mode: skipping small-n stages."
            echo "  Resuming from: $tsptwd_model"

            run_stage "[$s] Generate datasets n=200,300,500 with stored NN tour (fresh seed=$SEED)" \
                "$PYTHON ../../generate_train_dataset.py --sizes 200 300 500 --nn2opt \
                    --seed $SEED --out_dir ../../datasets/train"

            run_stage "[$s] TSPTWD FineTune — n_min=50 n_max=500, nn labels, JSON source, 5000 steps" \
                "$PYTHON train.py --size $s --mode tsptwd \
                    --resume $tsptwd_model \
                    --n_min 50 --n_max 500 --label nn --steps 5000 --lr 1e-4 \
                    --source tsptwd_json \
                    --out $tsptwd_model"

            run_stage "[$s] TSPTWD FineTune — n_min=300 n_max=600, nn labels, JSON source, 3000 steps" \
                "$PYTHON train.py --size $s --mode tsptwd \
                    --resume $tsptwd_model \
                    --n_min 300 --n_max 600 --label nn --steps 3000 --lr 5e-5 \
                    --source tsptwd_json \
                    --out $tsptwd_model"
        else
            run_stage "[$s] Generate datasets n≤100 with nn2opt (fresh seed=$SEED)" \
                "$PYTHON ../../generate_train_dataset.py --sizes 10 20 50 100 --nn2opt \
                    --seed $SEED --out_dir ../../datasets/train"

            run_stage "[$s] Generate datasets n=200,300,500 with stored NN tour (fresh seed=$SEED)" \
                "$PYTHON ../../generate_train_dataset.py --sizes 200 300 500 --nn2opt \
                    --seed $SEED --out_dir ../../datasets/train"

            if [[ -f "$tsptwd_model" ]]; then
                echo ""
                echo "  Existing model found: $tsptwd_model"
                echo "  Skipping Stage 1 — fine-tuning from existing weights."
            else
                run_stage "[$s] TSPTWD Stage 1 — n=8, optimal labels (fresh start)" \
                    "$PYTHON train.py --size $s --mode tsptwd \
                        --n 8 --label optimal --steps 3000 --lr 1e-3 \
                        --source random \
                        --out $tsptwd_model"
            fi

            for n in 10 50 100; do
                run_stage "[$s] TSPTWD Stage 2 — n=$n, nn2opt labels, JSON source" \
                    "$PYTHON train.py --size $s --mode tsptwd \
                        --resume $tsptwd_model \
                        --n $n --label nn2opt --steps 3000 --lr 5e-4 \
                        --source tsptwd_json \
                        --out $tsptwd_model"
            done

            for n in 150 300 500; do
                run_stage "[$s] TSPTWD Stage 3 — n=$n, nn labels, JSON source" \
                    "$PYTHON train.py --size $s --mode tsptwd \
                        --resume $tsptwd_model \
                        --n $n --label nn --steps 3000 --lr 1e-4 \
                        --source tsptwd_json \
                        --out $tsptwd_model"
            done

            if [[ "$XL" == "1" ]]; then
                echo "  WARNING: Stage 4 needs significant VRAM (O(n^2) edge tensor)."
                for n in 1000 3000 5000; do
                    run_stage "[$s] TSPTWD Stage 4 — n=$n, nn labels, JSON source" \
                        "$PYTHON train.py --size $s --mode tsptwd \
                            --resume $tsptwd_model \
                            --n $n --label nn --steps 300 --lr 5e-5 \
                            --source tsptwd_json \
                            --out $tsptwd_model"
                done
            fi
        fi

        echo ""
        echo "  Done: $tsptwd_model"
    fi
}

echo ""
echo "GNN Training Script (TSP + TSPTW-D)"
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
    *) echo "Usage: $0 [small|medium|large|all] [--tsptwd] [--xl]" >&2; exit 1 ;;
esac

echo ""
echo "All training complete."
