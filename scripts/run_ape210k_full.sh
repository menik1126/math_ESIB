#!/bin/bash
# ESIB Full Model (CN + SN) on APE210K
# Usage: bash scripts/run_ape210k_full.sh

export CUDA_VISIBLE_DEVICES=0

python run_seq2tree_APE_early_SP_VAE.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --hidden_size 1024 \
    --embedding_size 1024 \
    --n_epochs 50 \
    --beam_size 5 \
    --dropout 0.5 \
    --model_name roberta \
    --use_ape true

# Hyperparameters:
#   --batch_size      : Training batch size (default: 16)
#   --learning_rate   : Learning rate for AdamW optimizer (default: 5e-5)
#   --hidden_size     : Hidden layer dimension (default: 1024)
#   --embedding_size  : Embedding dimension (default: 1024)
#   --n_epochs        : Number of training epochs (default: 50 for APE210K)
#   --beam_size       : Beam search size for decoding (default: 5)
#   --dropout         : Dropout rate (default: 0.5)
#   --model_name      : Pretrained model: roberta | bert-base-chinese (default: roberta)
#   --use_ape         : Use APE210K dataset: true | false (default: true)



