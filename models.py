import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LayerNorm

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
            
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
            self.norms.append(LayerNorm(out_dim))
            
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, heads=8):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer with multiple heads
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
        self.norms.append(LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
            self.norms.append(LayerNorm(hidden_dim))
        
        # Final layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, out_dim, heads=1, dropout=dropout, concat=False))
            self.norms.append(LayerNorm(out_dim))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.norms.append(LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
            
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, out_dim))
            self.norms.append(LayerNorm(out_dim))
            
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class KGModel(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim, num_layers, dropout, gnn_type, device):
        super().__init__()
        self.device = device
        self.num_entities = num_entities
        self.num_relations = num_relations
        
        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, hidden_dim)
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # GNN Encoder
        if gnn_type == 'gcn':
            self.encoder = GCNEncoder(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        elif gnn_type == 'gat':
            self.encoder = GATEncoder(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        elif gnn_type == 'sage':
            self.encoder = SAGEEncoder(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")
        
        # Additional transformation with residual connection
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.to(device)

    def forward(self, edge_index):
        x = self.entity_embeddings.weight
        x_original = x
        x = self.encoder(x, edge_index)
        x = self.transform(x)
        # Residual connection
        x = x + x_original
        return x
    
    def get_entity_embeddings(self, edge_index):
        return self.forward(edge_index)
    
    def score(self, head_embs, rel_embs, tail_embs):
        """DistMult scoring function"""
        return torch.sum(head_embs * rel_embs * tail_embs, dim=-1)
    
    def predict(self, head_idx, rel_idx, edge_index):
        """Predict tail scores for given head and relation"""
        entity_embs = self.forward(edge_index)
        head_emb = entity_embs[head_idx]
        rel_emb = self.relation_embeddings(rel_idx)
        scores = torch.mm((head_emb * rel_emb).unsqueeze(0), entity_embs.t())
        return scores.squeeze(0)