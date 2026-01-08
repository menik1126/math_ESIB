# ESIB Training Scripts

本文件夹包含用于启动不同实验配置的 shell 脚本。

## Quick Start

```bash
cd math_ESIB

# Math23K Full Model
bash scripts/run_math23k_full.sh

# APE210K Full Model  
bash scripts/run_ape210k_full.sh
```

---

## Script Details

### 1. `run_math23k_full.sh` - Math23K 完整模型

**描述**: ESIB 完整模型，包含 CN (Core Network) + SN (Collaborator Network) + VAE

**Python 文件**: `run_seq2tree_bert_ultimate_divide_epoch_vae.py`

**用途**: 在 Math23K 数据集上训练完整的 ESIB 模型

```bash
bash scripts/run_math23k_full.sh
```

**默认参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 16 | 批次大小 |
| `learning_rate` | 5e-5 | 学习率 |
| `hidden_size` | 1024 | 隐藏层维度 |
| `embedding_size` | 1024 | 嵌入层维度 |
| `n_epochs` | 80 | 训练轮数 |
| `model_name` | roberta | 预训练模型 |

---

### 2. `run_ape210k_full.sh` - APE210K 完整模型

**描述**: ESIB 完整模型，在 APE210K 数据集上训练，支持早停

**Python 文件**: `run_seq2tree_APE_early_SP_VAE.py`

**用途**: 在 APE210K 数据集上训练完整的 ESIB 模型

```bash
bash scripts/run_ape210k_full.sh
```

**默认参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 16 | 批次大小 |
| `learning_rate` | 5e-5 | 学习率 |
| `hidden_size` | 1024 | 隐藏层维度 |
| `n_epochs` | 50 | 训练轮数 (APE210K 使用 50) |
| `model_name` | roberta | 预训练模型 |

---

### 3. `run_math23k_no_mt.sh` - 消融实验: 无多任务学习

**描述**: 消融实验 - 移除协作网络 (SN)，仅使用单网络 + VAE

**Python 文件**: `run_seq2tree_bert_ultimate_comp_vae.py`

**用途**: 验证多任务学习 (双网络结构) 的有效性

```bash
bash scripts/run_math23k_no_mt.sh
```

**对比**: 与 `run_math23k_full.sh` 相比，移除了 Collaborator Network (SN)

---

### 4. `run_math23k_no_vib.sh` - 消融实验: 无 VIB

**描述**: 消融实验 - 移除变分信息瓶颈 (VIB)，保留双网络结构

**Python 文件**: `run_seq2tree_bert_ultimate_divide_epoch.py`

**用途**: 验证 VIB 组件的有效性

```bash
bash scripts/run_math23k_no_vib.sh
```

**对比**: 与 `run_math23k_full.sh` 相比，移除了 VAE/VIB 组件

---

### 5. `run_math23k_baseline.sh` - 基线模型

**描述**: 基线模型 - 同时移除多任务学习和 VIB

**Python 文件**: `run_seq2tree_bert_ultimate_comp.py`

**用途**: 作为消融实验的基线对照

```bash
bash scripts/run_math23k_baseline.sh
```

**对比**: 基础的 Seq2Tree + RoBERTa 模型，无 MT 和 VIB

---

### 6. `run_math23k_dice.sh` - Dice Loss 变体

**描述**: 使用 Dice Loss 进行自蒸馏的变体模型

**Python 文件**: `run_seq2tree_bert_ultimate_divide_dice.py`

**用途**: 测试 Dice Loss (V_sdl) 替代标准交叉熵损失的效果

```bash
bash scripts/run_math23k_dice.sh
```

**特点**: 包含 CEB (Conditional Entropy Bottleneck) 模块

---

### 7. `run_math23k_bert.sh` - BERT 编码器

**描述**: 使用 BERT-base-chinese 替代 RoBERTa 作为编码器

**Python 文件**: `run_seq2tree_bert_ultimate_comp_vae.py`

**用途**: 对比不同预训练模型的效果

```bash
bash scripts/run_math23k_bert.sh
```

**默认参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 768 | BERT-base 隐藏层维度 |
| `embedding_size` | 768 | BERT-base 嵌入层维度 |
| `model_name` | bert-base-chinese | 预训练模型 |

---

## 命令行参数说明

所有脚本支持以下命令行参数:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch_size` | int | 16 | 训练批次大小 |
| `--learning_rate` | float | 5e-5 | AdamW 优化器学习率 |
| `--hidden_size` | int | 1024 | 隐藏层维度 (BERT 为 768) |
| `--embedding_size` | int | 1024 | 嵌入层维度 (BERT 为 768) |
| `--n_epochs` | int | 80/50 | 训练轮数 |
| `--beam_size` | int | 5 | Beam search 解码大小 |
| `--dropout` | float | 0.5 | Dropout 比例 |
| `--model_name` | str | roberta | 预训练模型名称 |
| `--use_ape` | str | false | 是否使用 APE210K 数据集 |
| `--warmup_period` | int | 3000 | 学习率 warmup 步数 |
| `--contra_weight` | float | 0.005 | 对比学习损失权重 |
| `--test_interval` | int | 5 | 测试间隔 (epochs) |

## 自定义训练示例

```bash
# 修改 batch_size 和 learning_rate
python run_seq2tree_bert_ultimate_divide_epoch_vae.py \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --n_epochs 100

# 使用 BERT 模型训练完整 ESIB
python run_seq2tree_bert_ultimate_divide_epoch_vae.py \
    --model_name bert-base-chinese \
    --hidden_size 768 \
    --embedding_size 768
```

## GPU 选择

```bash
# 单 GPU
export CUDA_VISIBLE_DEVICES=0

# 多 GPU (如果支持)
export CUDA_VISIBLE_DEVICES=0,1
```

## 实验对照表

| 脚本 | CN | SN | VIB | 数据集 | 说明 |
|------|:--:|:--:|:---:|--------|------|
| `run_math23k_full.sh` | ✓ | ✓ | ✓ | Math23K | 完整模型 |
| `run_ape210k_full.sh` | ✓ | ✓ | ✓ | APE210K | 完整模型 |
| `run_math23k_no_mt.sh` | ✓ | ✗ | ✓ | Math23K | 无多任务 |
| `run_math23k_no_vib.sh` | ✓ | ✓ | ✗ | Math23K | 无 VIB |
| `run_math23k_baseline.sh` | ✓ | ✗ | ✗ | Math23K | 基线 |
| `run_math23k_dice.sh` | ✓ | ✓ | ✓ | Math23K | Dice Loss |
| `run_math23k_bert.sh` | ✓ | ✗ | ✓ | Math23K | BERT 编码器 |

> **CN**: Core Network (核心网络)  
> **SN**: Collaborator Network (协作网络)  
> **VIB**: Variational Information Bottleneck (变分信息瓶颈)
