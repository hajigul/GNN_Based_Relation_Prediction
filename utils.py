import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class KGDataset(Dataset):
    def __init__(self, triples, num_ents, num_rels):
        self.triples = triples
        self.num_ents = num_ents
        self.num_rels = num_rels

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return torch.tensor(h, dtype=torch.long), torch.tensor(r, dtype=torch.long), torch.tensor(t, dtype=torch.long)

def collate_fn(batch):
    heads = torch.stack([item[0] for item in batch])
    rels = torch.stack([item[1] for item in batch])
    tails = torch.stack([item[2] for item in batch])
    return heads, rels, tails

def load_dataset(dataset_name, data_dir):
    dataset_path = os.path.join(data_dir, dataset_name)

    def load_triples(filename):
        triples = []
        with open(os.path.join(dataset_path, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, r, t = parts
                    triples.append((h, r, t))
        return triples

    train_triples = load_triples('train.txt')
    valid_triples = load_triples('valid.txt')
    test_triples = load_triples('test.txt')

    entities = set()
    relations = set()
    for h, r, t in train_triples + valid_triples + test_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    num_entities = len(entities)
    num_relations = len(relations)

    def convert(triples):
        return [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

    train_ids = convert(train_triples)
    valid_ids = convert(valid_triples)
    test_ids = convert(test_triples)

    return num_entities, num_relations, train_ids, valid_ids, test_ids, entity2id, relation2id

def build_graph(train_triples, num_entities):
    """Build bidirectional graph for message passing"""
    edge_index = []
    edge_type = []
    
    for h, r, t in train_triples:
        # Forward edge
        edge_index.append([h, t])
        edge_type.append(r)
        # Backward edge with reverse relation
        edge_index.append([t, h])
        edge_type.append(r + num_entities)  # Use separate ID for reverse edges
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return edge_index, edge_type

def create_dataloader(triples, num_ents, num_rels, batch_size, shuffle=True):
    dataset = KGDataset(triples, num_ents, num_rels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4)

def format_metrics(mr, mrr, hits1, hits3, hits10):
    """Format metrics with 4 decimal places for MRR and 2 for others"""
    return {
        'MR': f"{mr:.2f}",
        'MRR': f"{mrr:.4f}",
        'Hits@1': f"{hits1:.2f}",
        'Hits@3': f"{hits3:.2f}",
        'Hits@10': f"{hits10:.2f}"
    }