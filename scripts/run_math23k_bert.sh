#!/bin/bash
# ESIB with BERT encoder on Math23K
# Usage: bash scripts/run_math23k_bert.sh

export CUDA_VISIBLE_DEVICES=0

python run_seq2tree_bert_ultimate_comp_vae.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --hidden_size 768 \
    --embedding_size 768 \
    --n_epochs 80 \
    --beam_size 5 \
    --dropout 0.5 \
    --model_name bert-base-chinese \
    --use_ape false

# Using BERT instead of RoBERTa as encoder
# Note: hidden_size and embedding_size are 768 for BERT

