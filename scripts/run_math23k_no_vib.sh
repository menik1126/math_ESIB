#!/bin/bash
# ESIB Ablation: CN without VIB (w/o VIB) on Math23K
# Usage: bash scripts/run_math23k_no_vib.sh

export CUDA_VISIBLE_DEVICES=0

python run_seq2tree_bert_ultimate_divide_epoch.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --hidden_size 1024 \
    --embedding_size 1024 \
    --n_epochs 80 \
    --beam_size 5 \
    --dropout 0.5 \
    --model_name roberta \
    --use_ape false

# Ablation Study: Without Variational Information Bottleneck
# This script runs the model without the VIB component



