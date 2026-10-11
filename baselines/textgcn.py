"""
TextGCN: Text Graph Convolutional Network
Reference: Yao et al., "Graph Convolutional Networks for Text Classification"
           (AAAI 2019), https://arxiv.org/abs/1809.05679

TextGCN builds a heterogeneous word-document graph from text corpus and
applies GCN for classification. In the original paper, the graph contains:
  - Document nodes (connected to their words via edges)
  - Word nodes (connected via PMI-based co-occurrence edges)

For our TAG (text-attributed graph) setting, we adapt TextGCN as follows:
  - Keep the original graph topology (document/document-node connections)
  - Add word nodes for each document's text, connected to their document
  - Add PMI-based word-word edges for words that co-occur in the same window
  - Run GCN on this heterogeneous text-graph for node classification

This faithfully reproduces the TextGCN paradigm: using text-derived graph
structure as the primary signal for classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Graph Construction: Build TextGCN's heterogeneous word-document graph
# ──────────────────────────────────────────────────────────────────────────────

def build_textgcn_graph(
    G,
    labels,
    text_map,
    window_size=20,
    min_word_freq=2,
    max_vocab_size=2000,
):
    """Build TextGCN's heterogeneous graph from an existing TAG.

    The resulting graph has:
      - Original graph nodes (document nodes)
      - Word nodes (one per unique word in corpus)
      - Edges:
          1. Original graph edges (doc-doc)
          2. Doc-word edges (each doc connected to its words, weight=1)
          3. Word-word edges (PMI-based co-occurrence)

    Args:
        G: networkx Graph with node features
        labels: list/array of node labels
        text_map: dict mapping node_id -> text string
        window_size: sliding window size for PMI computation
        min_word_freq: minimum word frequency to include in vocab
        max_vocab_size: maximum vocabulary size (for memory)

    Returns:
        adj: numpy array [N_total, N_total] (normalized adjacency)
        features: numpy array [N_total, D] (one-hot style features)
        num_doc_nodes: int (number of original graph nodes)
        word_vocab: dict mapping word -> index
    """
    num_doc_nodes = G.number_of_nodes()
    node_list = list(G.nodes())

    # ── Step 1: Build vocabulary from corpus ──
    word_freq = defaultdict(int)
    doc_words = {}  # node_id -> list of words

    for node_id in node_list:
        text = text_map.get(node_id, "")
        if not text or text == "No description available.":
            words = []
        else:
            words = text.lower().split()[:window_size]
        doc_words[node_id] = words
        for w in words:
            word_freq[w] += 1

    # Filter vocabulary
    vocab_items = sorted(
        [(w, f) for w, f in word_freq.items() if f >= min_word_freq],
        key=lambda x: -x[1],
    )[:max_vocab_size]

    word_vocab = {w: idx for idx, (w, _) in enumerate(vocab_items)}
    num_word_nodes = len(word_vocab)
    num_total = num_doc_nodes + num_word_nodes

    # ── Step 2: Compute PMI for word-word edges ──
    # PMI(i,j) = log(p(i,j) / (p(i) * p(j)))
    # where p(i,j) = #windows containing both i,j / #total_windows
    #       p(i)   = #windows containing i / #total_windows
    word_window_count = defaultdict(int)
    pair_window_count = defaultdict(int)
    total_windows = 0

    for node_id in node_list:
        words = [w for w in doc_words[node_id] if w in word_vocab]
        # Each document is treated as one window
        total_windows += 1
        unique_words = set(words)
        for w in unique_words:
            word_window_count[w] += 1
        for w1 in unique_words:
            for w2 in unique_words:
                if w1 < w2:
                    pair_window_count[(w1, w2)] += 1

    # Compute PMI values
    pmi_threshold = 0.0  # Only keep positive PMI (positive correlation)
    word_word_edges = []

    for (w1, w2), count in pair_window_count.items():
        p_ij = count / total_windows
        p_i = word_window_count[w1] / total_windows
        p_j = word_window_count[w2] / total_windows
        pmi = np.log(p_ij / (p_i * p_j + 1e-10) + 1e-10)
        if pmi > pmi_threshold:
            i = num_doc_nodes + word_vocab[w1]
            j = num_doc_nodes + word_vocab[w2]
            word_word_edges.append((i, j, pmi))

    # ── Step 3: Build adjacency matrix ──
    adj = np.zeros((num_total, num_total), dtype=np.float32)

    # (a) Original graph edges (doc-doc)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # (b) Doc-word edges (weight = 1 for each word occurrence)
    for node_id in node_list:
        doc_idx = node_to_idx[node_id]
        words = doc_words[node_id]
        for w in words:
            if w in word_vocab:
                word_idx = num_doc_nodes + word_vocab[w]
                adj[doc_idx, word_idx] = 1.0
                adj[word_idx, doc_idx] = 1.0

    # (c) Word-word edges (PMI-weighted)
    for i, j, pmi in word_word_edges:
        adj[i, j] = pmi
        adj[j, i] = pmi

    # (d) Add self-loops
    adj += np.eye(num_total, dtype=np.float32)

    # (e) Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt

    # ── Step 4: Build feature matrix ──
    # TextGCN uses one-hot features: each node has a unique feature dimension
    # For efficiency, we use a low-dimensional projection instead
    # Document nodes: use original graph features if available
    # Word nodes: use learned embeddings (handled by the model)

    # For doc nodes: use original features
    sample_node = node_list[0]
    feat_dim = len(G.nodes[sample_node]["feat"]) if "feat" in G.nodes[sample_node] else 0

    if feat_dim > 0:
        doc_features = np.zeros((num_doc_nodes, feat_dim), dtype=np.float32)
        for i, n in enumerate(node_list):
            doc_features[i] = G.nodes[n]["feat"]
    else:
        # Fallback: one-hot
        feat_dim = num_doc_nodes
        doc_features = np.eye(num_doc_nodes, dtype=np.float32)

    # Word nodes: zero features (will use learned embeddings in model)
    word_features = np.zeros((num_word_nodes, feat_dim), dtype=np.float32)
    features = np.vstack([doc_features, word_features])

    return adj_normalized, features, num_doc_nodes, word_vocab


# ──────────────────────────────────────────────────────────────────────────────
# TextGCN Model: 2-layer GCN on the heterogeneous text graph
# ──────────────────────────────────────────────────────────────────────────────

class TextGCN(nn.Module):
    """TextGCN: 2-layer GCN on word-document heterogeneous graph.

    Following the original paper's architecture:
      H1 = ReLU(A_hat @ X @ W0)
      H2 = softmax(A_hat @ H1 @ W1)

    We add a word embedding layer for word nodes that have zero initial features.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        label_dim,
        num_word_nodes=0,
        word_embed_dim=64,
        dropout=0.5,
    ):
        super().__init__()
        self.num_word_nodes = num_word_nodes
        self.word_embed_dim = word_embed_dim

        # Word embedding (for word nodes with zero initial features)
        if num_word_nodes > 0 and word_embed_dim > 0:
            self.word_embedding = nn.Embedding(num_word_nodes, word_embed_dim)
            # Adjust input_dim to accommodate word embeddings
            effective_input_dim = input_dim + word_embed_dim
        else:
            self.word_embedding = None
            effective_input_dim = input_dim

        # 2-layer GCN (as in original TextGCN paper)
        self.W0 = nn.Linear(effective_input_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, label_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def forward(self, features, adj, num_doc_nodes):
        """
        Args:
            features: [N_total, D]  (doc features + word features)
            adj:      [N_total, N_total] (normalized adjacency)
            num_doc_nodes: int

        Returns:
            pred: [num_doc_nodes, label_dim]  (predictions for doc nodes only)
        """
        N_total = features.shape[0]

        # Augment features with word embeddings for word nodes
        if self.word_embedding is not None and self.num_word_nodes > 0:
            word_indices = torch.arange(self.num_word_nodes, device=features.device)
            word_embs = self.word_embedding(word_indices)  # [num_words, word_embed_dim]

            # Pad doc features with zeros, pad word features with embeddings
            doc_pad = torch.zeros(num_doc_nodes, self.word_embed_dim, device=features.device)
            x = torch.cat([
                torch.cat([features[:num_doc_nodes], doc_pad], dim=1),
                torch.cat([features[num_doc_nodes:], word_embs], dim=1),
            ], dim=0)
        else:
            x = features

        # Layer 1: H1 = ReLU(A @ X @ W0)
        x = torch.matmul(adj, x)  # [N_total, effective_input_dim]
        x = self.W0(x)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 2: H2 = A @ H1 @ W1
        x = torch.matmul(adj, x)  # [N_total, hidden_dim]
        x = self.W1(x)

        # Return predictions for document nodes only
        return x[:num_doc_nodes]


# ──────────────────────────────────────────────────────────────────────────────
# Text loading utility
# ──────────────────────────────────────────────────────────────────────────────

def load_text_map(dataset_name, data_dir="data"):
    """Load node texts for a dataset.

    Returns:
        text_map: dict mapping node_id -> text string
    """
    text_dir = Path(data_dir) / "texts" / dataset_name
    text_map = {}

    if not text_dir.exists():
        return text_map

    for fname in os.listdir(text_dir):
        if not fname.endswith(".txt"):
            continue
        if not fname.startswith("node_"):
            continue
        try:
            node_id = int(fname.split(".")[0].split("_")[1])
        except Exception:
            continue
        with open(text_dir / fname, "r", encoding="utf-8") as f:
            text_map[node_id] = f.read().strip()

    return text_map

