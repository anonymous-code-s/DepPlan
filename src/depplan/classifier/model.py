"""Strategy GCN with projected BGE features, mean pooling and an MLP head."""
import torch
from torch import nn

LABELS = ('sequential', 'parallel', 'hierarchical')


def graph_structure(triples):
    concepts = {s: t for s, r, t in triples if r == ':instance'}
    nodes = list(concepts)
    index = {v: i for i, v in enumerate(nodes)}
    adjacency = torch.eye(len(nodes))
    for s, r, t in triples:
        if r != ':instance' and s in index and t in index:
            adjacency[index[s], index[t]] = 1
            adjacency[index[t], index[s]] = 1
    degree = adjacency.sum(1).rsqrt()
    adjacency = degree[:, None] * adjacency * degree[None, :]
    return [concepts[v] for v in nodes], adjacency


def collate(graphs, device):
    features = torch.cat([g['x'] for g in graphs]).to(device)
    adjacency = torch.block_diag(*[g['adj'] for g in graphs]).to(device)
    membership = torch.cat([torch.full((len(g['x']),), i, dtype=torch.long)
                            for i, g in enumerate(graphs)]).to(device)
    labels = torch.tensor([g.get('y', 0) for g in graphs], device=device)
    return features, adjacency, membership, labels


class StrategyGCN(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.projection = nn.Linear(input_dim, 128)
        self.gcn1 = nn.Linear(128, 128)
        self.gcn2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(self, x, adj, membership):
        x = self.projection(x)
        x = self.dropout(torch.relu(self.gcn1(adj @ x)))
        x = self.dropout(torch.relu(self.gcn2(adj @ x)))
        pooled = x.new_zeros((int(membership.max()) + 1, x.shape[1]))
        pooled.index_add_(0, membership, x)
        pooled /= torch.bincount(membership).to(x.dtype)[:, None]
        return self.mlp(pooled)
