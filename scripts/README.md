# ESIB Training Scripts

## Quick Start

```bash
# Math23K Full Model
bash scripts/run_math23k_full.sh

# APE210K Full Model  
bash scripts/run_ape210k_full.sh
```

## Available Scripts

| Script | Description | Dataset |
|--------|-------------|---------|
| `run_math23k_full.sh` | Full ESIB model (CN + SN) | Math23K |
| `run_ape210k_full.sh` | Full ESIB model (CN + SN) | APE210K |
| `run_math23k_no_mt.sh` | Ablation: w/o Multi-Task | Math23K |
| `run_math23k_no_vib.sh` | Ablation: w/o VIB | Math23K |
| `run_math23k_baseline.sh` | Ablation: w/o MT+VIB | Math23K |
| `run_math23k_dice.sh` | Variant: with Dice Loss | Math23K |
| `run_math23k_bert.sh` | Using BERT encoder | Math23K |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 5e-5 | Learning rate for AdamW optimizer |
| `hidden_size` | 1024 | Hidden layer dimension (768 for BERT) |
| `embedding_size` | 1024 | Embedding dimension (768 for BERT) |
| `n_epochs` | 80/50 | Training epochs (80 for Math23K, 50 for APE210K) |
| `beam_size` | 5 | Beam search size for decoding |
| `dropout` | 0.5 | Dropout rate |
| `model_name` | roberta | Encoder: `roberta` or `bert-base-chinese` |
| `use_ape` | false | Dataset: `true` for APE210K, `false` for Math23K |

## Advanced Config (src/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contra_weight` | 0.005 | Contrastive loss weight |
| `CEB_weight` | 0.005 | Conditional Entropy Bottleneck weight |
| `latent_dim` | 50 | VAE latent dimension |
| `warmup_period` | 3000 | Learning rate warmup steps |
| `test_interval` | 5 | Evaluation interval (epochs) |
| `warm_up_stratege` | original | Warmup strategy: `original` or `LinearWarmup` |
| `RDloss` | kl_loss | Loss type: `kl_loss`, `cosine_loss`, etc. |

## GPU Selection

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Multiple GPUs (if supported)
export CUDA_VISIBLE_DEVICES=0,1
```

## Example: Custom Training

```bash
export CUDA_VISIBLE_DEVICES=0

# Modify config.py first, then run:
python run_seq2tree_bert_ultimate_divide_epoch_vae.py
```

