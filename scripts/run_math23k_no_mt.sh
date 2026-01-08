#!/bin/bash
# ESIB Ablation: CN without Multi-Task (w/o MT) on Math23K
# Usage: bash scripts/run_math23k_no_mt.sh

export CUDA_VISIBLE_DEVICES=0

python run_seq2tree_bert_ultimate_comp_vae.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --hidden_size 1024 \
    --embedding_size 1024 \
    --n_epochs 80 \
    --beam_size 5 \
    --dropout 0.5 \
    --model_name roberta \
    --use_ape false

# Ablation Study: Without Multi-Task Learning
# This script runs the model without the collaborator network (SN)

