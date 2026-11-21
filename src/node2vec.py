"""Node2Vec structural embeddings (biased random walks + skip-gram with negative sampling) in torch.

gensim has no Python 3.14 wheel, so the skip-gram step is implemented here directly.
"""
from __future__ import annotations

import random

import networkx as nx
import numpy as np
import torch


def _walks(G: nx.Graph, idx: dict, num_walks: int, walk_len: int, p: float, q: float, rng: random.Random):
    nodes = list(G.nodes)
    adj = {n: list(G.neighbors(n)) for n in nodes}
    nbr_set = {n: set(adj[n]) for n in nodes}
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_len:
                cur = walk[-1]
                nbrs = adj[cur]
                if not nbrs:
                    break
                if len(walk) == 1:
                    walk.append(rng.choice(nbrs))
                    continue
                prev = walk[-2]
                weights = []
                for x in nbrs:
                    if x == prev:
                        weights.append(1.0 / p)
                    elif x in nbr_set[prev]:
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / q)
                walk.append(rng.choices(nbrs, weights=weights, k=1)[0])
            yield [idx[n] for n in walk]


def node2vec_embeddings(
    G: nx.MultiDiGraph,
    dim: int = 64,
    num_walks: int = 6,
    walk_len: int = 15,
    window: int = 3,
    p: float = 1.0,
    q: float = 0.5,
    epochs: int = 4,
    negatives: int = 5,
    seed: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Return (node_order, matrix[n, dim]) with L2-normalised rows."""
    order = list(G.nodes)
    n = len(order)
    if n == 0:
        return order, np.zeros((0, dim), dtype=np.float32)
    idx = {node: i for i, node in enumerate(order)}
    U = nx.Graph(G.to_undirected(as_view=False))
    U.add_nodes_from(order)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    pairs: list[tuple[int, int]] = []
    for walk in _walks(U, idx, num_walks, walk_len, p, q, rng):
        for i, c in enumerate(walk):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if j != i:
                    pairs.append((c, walk[j]))
    emb = torch.nn.Embedding(n, dim)
    ctx = torch.nn.Embedding(n, dim)
    torch.nn.init.uniform_(emb.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(ctx.weight)
    if pairs:
        deg = torch.tensor([max(U.degree(node), 1) for node in order], dtype=torch.float) ** 0.75
        noise = deg / deg.sum()
        data = torch.tensor(pairs, dtype=torch.long)
        opt = torch.optim.Adam(list(emb.parameters()) + list(ctx.parameters()), lr=0.01)
        for _ in range(epochs):
            perm = torch.randperm(len(data))
            for s in range(0, len(data), 1024):
                b = data[perm[s : s + 1024]]
                c, o = emb(b[:, 0]), ctx(b[:, 1])
                pos = torch.nn.functional.logsigmoid((c * o).sum(-1))
                neg_ids = torch.multinomial(noise, len(b) * negatives, replacement=True).view(len(b), negatives)
                neg = torch.nn.functional.logsigmoid(-(ctx(neg_ids) @ c.unsqueeze(-1)).squeeze(-1)).sum(-1)
                loss = -(pos + neg).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    mat = emb.weight.detach().numpy().astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-9, None)
    return order, mat
