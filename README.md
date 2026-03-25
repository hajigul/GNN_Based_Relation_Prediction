# GNN-Based Relation Prediction on Knowledge Graphs

Implements **GCN**, **GAT**, and **GraphSAGE** for **relation prediction**
(predicting the relation *r* between head entity *h* and tail entity *t*) on
five standard KG benchmarks.

# Knowledge Graph Relation Prediction with Graph Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4.0-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive implementation of Graph Neural Networks (GCN, GAT, and GraphSAGE) for knowledge graph relation prediction. This repository provides end-to-end training and evaluation pipelines for multiple benchmark datasets with automatic sequential execution.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architectures](#model-architectures)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements three state-of-the-art Graph Neural Network architectures for predicting relations in knowledge graphs:

- **GCN (Graph Convolutional Networks)**: Captures local neighborhood information through graph convolutions
- **GAT (Graph Attention Networks)**: Uses attention mechanisms to weight neighbor importance
- **GraphSAGE (Graph Sample and Aggregator)**: Samples and aggregates features from local neighborhoods

The models use the DistMult scoring function and are trained with self-adversarial negative sampling to achieve competitive results on standard knowledge graph completion benchmarks.

## Features

- **Multi-GPU Support**: Automatic detection and utilization of multiple GPUs for faster training
- **Sequential Dataset Processing**: Run training on multiple datasets automatically
- **Memory Optimization**: Dataset-specific hyperparameter tuning for large datasets (YAGO3-10)
- **Comprehensive Evaluation**: MR, MRR, Hits@1, Hits@3, Hits@10 metrics with decimal formatting
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Learning Rate Scheduling**: ReduceLROnPlateau for optimal convergence
- **Gradient Clipping**: Prevents gradient explosion
- **Residual Connections**: Improves gradient flow in deep networks
- **Layer Normalization**: Stabilizes training

## Model Architectures

### GCN (Graph Convolutional Network)
- Uses graph convolutions to aggregate neighborhood information
- Applies layer normalization and residual connections
- Efficient for capturing local graph structure

### GAT (Graph Attention Network)
- Employs multi-head attention mechanisms
- Learns importance weights for different neighbors
- Better at capturing complex relationships

### GraphSAGE
- Samples fixed-size neighborhoods
- Aggregates features through various functions (mean, max)
- Scalable to large graphs



---

## Project Structure

```
GNN_Based_relation_pre/
├── src/
│   ├── config.py      # All hyperparameters via argparse
│   ├── model.py       # GCN / GAT / GraphSAGE model classes
│   ├── utils.py       # Data loading, negative sampling, evaluation
│   ├── train.py       # Training loop + checkpointing
│   ├── evaluate.py    # Standalone evaluation script
│   └── run_all.py     # Batch runner for all model × dataset combos
├── data/
│   ├── FB15k-237/     train.txt  valid.txt  test.txt  [entities.txt  relations.txt]
│   ├── WN18/
│   ├── WN18RR/
│   ├── FB15k/
│   └── YAGO3-10/
├── results/           # Checkpoints + logs (auto-created)
├── gnn_env/           # Virtual environment (created by setup.sh)
├── requirements.txt
└── setup.sh           # One-shot setup + smoke test
```

---

## Quick Start

### 1  Set up environment & run a smoke test

```bash
cd /home/user/23h1710_KGC/GNN_Based_relation_pre
bash setup.sh
```

This will:
- Create a virtual environment `gnn_env/`
- Install PyTorch (CPU) + NumPy
- Run a 50-epoch GCN smoke test on FB15k-237

### 2  Activate environment manually

```bash
source /home/user/23h1710_KGC/GNN_Based_relation_pre/gnn_env/bin/activate
cd /home/user/23h1710_KGC/GNN_Based_relation_pre/src
```

### 3  Train a single model

```bash

# Run GCN on all datasets with multi-GPU, it will provide results table. For GAT and GSAG change gcn to gat and gsag
python main.py --gnn_type gcn --use_multi_gpu


# More analysis, GCN on FB15k-237
python train.py --model GCN --dataset FB15k-237

# GAT on WN18RR  (4 attention heads)
python train.py --model GAT --dataset WN18RR --num_heads 4

# GraphSAGE on FB15k  (mean aggregator)
python train.py --model GraphSAGE --dataset FB15k --aggregator mean
```

### 4  Run all models on all datasets

```bash
python run_all.py --epochs 500
# Only specific combos:
python run_all.py --models GCN GAT --datasets FB15k-237 WN18RR --epochs 300
```

### 5  Evaluate a saved checkpoint

```bash
python evaluate.py \
    --model GCN --dataset FB15k-237 \
    --checkpoint ../results/GCN_FB15k-237/checkpoint_best.pt
```

---

## Metrics

All evaluation uses **filtered** settings (false negatives removed):


##  Supported Datasets

| Dataset | Entities | Relations | Train | Valid | Test |
|---------|----------|-----------|-------|-------|------|
| FB15k-237 | 14,265 | 237 | 272,115 | 17,535 | 20,466 |
| WN18 | 40,943 | 18 | 141,442 | 5,000 | 5,000 |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |
| YAGO3-10 | 123,182 | 37 | 1,079,040 | 5,000 | 5,000 |
| FB15k | 14,951 | 1,345 | 483,142 | 50,000 | 59,071 |


##  GNN-based Model Results

The following table presents the performance of our implemented GNN-based models across five standard knowledge graph completion datasets.

| **Dataset** | **Model** | **MR ↓** | **MRR ↑** | **Hits@1 ↑** | **Hits@3 ↑** | **Hits@10 ↑** |
|-------------|-----------|----------|-----------|--------------|--------------|---------------|
| **FB15k** | GCN | 70.03 | 0.1987 | 0.0940 | 0.2153 | 0.4089 |
| | GAT | 69.26 | 0.3284 | 0.1785 | 0.3996 | 0.6219 |
| | GraphSAGE | 217.73 | 0.1799 | 0.0936 | 0.2060 | 0.3483 |
| **WN18** | GCN | 5.48 | 0.3578 | 0.1582 | 0.4232 | 0.8678 |
| | GAT | 5.56 | 0.3219 | 0.1246 | 0.3646 | 0.8858 |
| | GraphSAGE | 4.04 | 0.4638 | 0.2316 | 0.6654 | 0.9108 |
| **FB15k-237** | GCN | 20.38 | 0.3655 | 0.2335 | 0.4115 | 0.6535 |
| | GAT | 10.40 | 0.5474 | 0.4169 | 0.6234 | 0.7874 |
| | GraphSAGE | 52.45 | 0.2670 | 0.1901 | 0.2857 | 0.4143 |
| **WN18RR** | GCN | 4.95 | 0.3231 | 0.1238 | 0.3363 | 0.9515 |
| | GAT | 4.48 | 0.3604 | 0.1452 | 0.4190 | 0.9828 |
| | GraphSAGE | 4.35 | 0.2949 | 0.0364 | 0.3781 | 0.9917 |
| **YAGO3-10** | GCN | 7.28 | 0.4904 | 0.3108 | 0.6052 | 0.7612 |
| | GAT | 6.31 | 0.4623 | 0.2838 | 0.5656 | 0.7852 |
| | GraphSAGE | 13.04 | 0.2736 | 0.1458 | 0.2974 | 0.5224 |

### Performance Highlights

- **Best Overall Performance**: GAT achieves the highest MRR (0.547) and Hits@10 (0.787) on FB15k-237
- **Best on WN18**: GraphSAGE achieves the best MRR (0.464) and Hits@10 (0.911)
- **Best on WN18RR**: GraphSAGE achieves the highest Hits@10 (0.992)
- **Best on YAGO3-10**: GCN achieves the best MRR (0.490) and Hits@10 (0.761)
- **Lowest Mean Rank**: GAT achieves the lowest MR (10) on FB15k-237, and multiple models achieve MR 4-5 on WN18 and WN18RR

### Key Observations

1. **GAT excels on dense datasets** (FB15k-237) with complex relation patterns
2. **GraphSAGE performs exceptionally well on word-based datasets** (WN18, WN18RR)
3. **GCN shows strong scalability** on large datasets (YAGO3-10) with 123K entities
4. **All GNN models achieve competitive Hits@10 scores** on WN18RR (0.95+)

### Training Configuration

Optimal hyperparameters used for each dataset:

| Dataset | Hidden Dim | Batch Size | Negative Samples | Learning Rate | Epochs |
|---------|------------|------------|------------------|---------------|--------|
| FB15k-237 | 256 | 512 | 128 | 0.001 | 200 |
| WN18 | 256 | 512 | 128 | 0.001 | 200 |
| WN18RR | 256 | 512 | 128 | 0.001 | 200 |
| YAGO3-10 | 128 | 64 | 32 | 0.001 | 200 |
| FB15k | 256 | 512 | 128 | 0.001 | 200 |


##  Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 12.1 (for GPU support)
- NVIDIA GPU with at least 8GB VRAM (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/knowledge-graph-gnn.git
cd knowledge-graph-gnn

---

## Model Details

### GCN  (Relational GCN)
- Basis decomposition to handle large relation sets
- Inverse edges added automatically
- 2 layers, 200-dim embeddings by default

### GAT  (Graph Attention Network)
- Multi-head attention (4 heads default)
- Relation-aware message passing
- Concat + projection to maintain output dimension

### GraphSAGE
- Supports three aggregators: `mean`, `max`, `lstm`
- L2 normalisation after each layer
- Concatenates ego + aggregated neighbour features

---

## Key Arguments

| Argument           | Default      | Description                              |
|--------------------|--------------|------------------------------------------|
| `--model`          | `GCN`        | `GCN` / `GAT` / `GraphSAGE`             |
| `--dataset`        | `FB15k-237`  | One of 5 supported datasets              |
| `--embedding_dim`  | `200`        | Entity/relation embedding size           |
| `--hidden_dim`     | `200`        | GNN hidden dimension                     |
| `--num_layers`     | `2`          | GNN depth                                |
| `--dropout`        | `0.3`        | Dropout probability                      |
| `--epochs`         | `500`        | Training epochs                          |
| `--batch_size`     | `1024`       | Training batch size                      |
| `--lr`             | `0.001`      | Adam learning rate                       |
| `--negative_samples`| `256`       | Negatives per positive triple            |
| `--num_heads`      | `4`          | GAT attention heads                      |
| `--aggregator`     | `mean`       | GraphSAGE aggregator                     |
| `--device`         | `auto`       | `auto` / `cpu` / `cuda`                 |
| `--eval_every`     | `10`         | Eval interval (epochs)                   |

---

## GPU Support

Edit `setup.sh` to install the CUDA build of PyTorch:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then run with `--device cuda`.
