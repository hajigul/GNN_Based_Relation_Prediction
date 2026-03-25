import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, num_relations, negative_sample_size, device, entity_embs, epoch, warmup_epochs):
    model.train()
    total_loss = 0
    
    # Handle DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        rel_embeddings = model.module.relation_embeddings.weight
    else:
        rel_embeddings = model.relation_embeddings.weight

    for heads, rels, tails in train_loader:
        heads = heads.to(device)
        rels = rels.to(device)
        tails = tails.to(device)
        batch_size = heads.size(0)
        
        # Get embeddings
        head_embs = entity_embs[heads]
        tail_embs = entity_embs[tails]
        pos_rel_embs = rel_embeddings[rels]
        
        # Positive scores
        if isinstance(model, torch.nn.DataParallel):
            pos_scores = model.module.score(head_embs, pos_rel_embs, tail_embs)
        else:
            pos_scores = model.score(head_embs, pos_rel_embs, tail_embs)
        
        # Negative sampling with corruption of tail or head
        neg_tails = torch.randint(0, entity_embs.size(0), (batch_size, negative_sample_size), device=device)
        neg_heads = torch.randint(0, entity_embs.size(0), (batch_size, negative_sample_size), device=device)
        
        # Mix corruption types (50% tail corruption, 50% head corruption)
        corrupt_mode = torch.rand(batch_size, device=device) < 0.5
        
        # Create negative samples
        head_neg = head_embs.unsqueeze(1).expand(-1, negative_sample_size, -1).clone()
        tail_neg = tail_embs.unsqueeze(1).expand(-1, negative_sample_size, -1).clone()
        
        # Replace head or tail with random entities
        head_neg[corrupt_mode] = entity_embs[neg_heads[corrupt_mode]]
        tail_neg[~corrupt_mode] = entity_embs[neg_tails[~corrupt_mode]]
        
        # Expand relation embeddings for negative samples
        rel_neg = pos_rel_embs.unsqueeze(1).expand(-1, negative_sample_size, -1)
        
        # Negative scores
        if isinstance(model, torch.nn.DataParallel):
            neg_scores = model.module.score(head_neg, rel_neg, tail_neg)
        else:
            neg_scores = model.score(head_neg, rel_neg, tail_neg)
        
        # Self-adversarial negative sampling weights
        neg_weights = F.softmax(neg_scores.detach(), dim=1)
        
        # Loss with adaptive temperature
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=1).mean()
        
        loss = pos_loss + neg_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, data_loader, entity_embs, num_relations, device):
    model.eval()
    ranks = []
    
    if isinstance(model, torch.nn.DataParallel):
        rel_embeddings = model.module.relation_embeddings.weight
    else:
        rel_embeddings = model.relation_embeddings.weight

    with torch.no_grad():
        for heads, rels, tails in tqdm(data_loader, desc="Evaluating"):
            heads = heads.to(device)
            rels = rels.to(device)
            tails = tails.to(device)
            batch_size = heads.size(0)
            
            head_embs = entity_embs[heads]
            tail_embs = entity_embs[tails]
            rel_embs = rel_embeddings[rels]
            
            # Compute scores for all relations
            head_tail = head_embs * tail_embs
            scores = torch.mm(head_tail, rel_embeddings.t())
            
            # Get rank
            target_scores = scores[torch.arange(batch_size), rels]
            rank = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(rank.cpu().numpy())
    
    # Compute metrics (all in 0-1 range, not percentages)
    ranks = np.array(ranks)
    mr = np.mean(ranks)
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)  # Already 0-1
    hits3 = np.mean(ranks <= 3)  # Already 0-1
    hits10 = np.mean(ranks <= 10)  # Already 0-1
    
    return mr, mrr, hits1, hits3, hits10