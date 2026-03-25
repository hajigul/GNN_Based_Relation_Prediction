# GNN-Based Relation Prediction on Knowledge Graphs

Implements **GCN**, **GAT**, and **GraphSAGE** for **relation prediction**
(predicting the relation *r* between head entity *h* and tail entity *t*) on
five standard KG benchmarks.

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
# GCN on FB15k-237
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

| Metric  | Description                          |
|---------|--------------------------------------|
| MR      | Mean Rank (lower is better)          |
| MRR     | Mean Reciprocal Rank (higher better) |
| Hits@1  | % of queries ranked in top 1         |
| Hits@3  | % of queries ranked in top 3         |
| Hits@10 | % of queries ranked in top 10        |

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
