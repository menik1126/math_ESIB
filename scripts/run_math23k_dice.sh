#!/bin/bash
# ESIB with Dice Loss (V_sdl) on Math23K
# Usage: bash scripts/run_math23k_dice.sh

export CUDA_VISIBLE_DEVICES=0

python run_seq2tree_bert_ultimate_divide_dice.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --hidden_size 1024 \
    --embedding_size 1024 \
    --n_epochs 80 \
    --beam_size 5 \
    --dropout 0.5 \
    --model_name roberta \
    --use_ape false

# Variant: Using Dice Loss for self-distillation
# This script uses V_sdl (Dice Loss) instead of standard cross-entropy

