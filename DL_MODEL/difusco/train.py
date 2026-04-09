"""
train.py — Training loop for DIFUSCO

TODO (pair-coding session):
  - build dataset + DataLoader
  - forward diffusion: add noise to ground-truth adjacency at timestep t
  - training objective: predict noise (or x_0) at each step
  - optimiser, LR schedule, gradient clipping
  - checkpoint saving to model/

Usage (once implemented):
    python train.py --n 10 --epochs 100 --batch 32 --size small
"""
