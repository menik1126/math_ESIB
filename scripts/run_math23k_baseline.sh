#!/bin/bash
# ESIB Ablation: Baseline without MT+VIB on Math23K
# Usage: bash scripts/run_math23k_baseline.sh

export CUDA_VISIBLE_DEVICES=0

python run_seq2tree_bert_ultimate_comp.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --hidden_size 1024 \
    --embedding_size 1024 \
    --n_epochs 80 \
    --beam_size 5 \
    --dropout 0.5 \
    --model_name roberta \
    --use_ape false

# Ablation Study: Baseline without both Multi-Task and VIB
# This is the base seq2tree model with RoBERTa encoder

