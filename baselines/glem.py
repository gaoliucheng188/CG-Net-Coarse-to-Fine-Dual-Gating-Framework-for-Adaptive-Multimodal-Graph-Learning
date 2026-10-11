"""
GLEM: Graph-Aware Language Model (Simplified Reproduction)
Reference: Zhao et al., "Graph-Aware Language Model (GLEM)"
           (NeurIPS 2023), https://arxiv.org/abs/2310.09833

GLEM uses an Expectation-Maximization (EM) iterative framework to jointly
train a GNN and a Language Model (LM) on text-attributed graphs:

  E-step: Use the current GNN/LM to generate pseudo-labels for each other
  M-step: Re-train GNN (with LM pseudo-labels) and LM (with GNN pseudo-labels)

Simplified implementation uses:
  - GNN: 3-layer GCN over graph + node features + (optionally) LM pseudo-labels
  - LM:  MLP over text embeddings from a lightweight text encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from multimodal.text_encoder import TextEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Simple GCN layer (avoids the BN dim issues of the project's GcnEncoderNode)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleGCNLayer(nn.Module):
    """Single GCN layer: H' = ReLU(A_norm @ H @ W)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        init.xavier_uniform_(self.weight, gain=init.calculate_gain("relu"))

    def forward(self, x, adj):
        # x: [B, N, D_in], adj: [B, N, N]
        h = torch.matmul(adj, x)             # [B, N, D_in]
        h = torch.matmul(h, self.weight)     # [B, N, D_out]
        if self.bias is not None:
            h = h + self.bias
        return h


class SimpleGCN(nn.Module):
    """3-layer GCN for GLEM. Input: [B, N, D] (B=1 for single-graph)."""

    def __init__(self, input_dim, hidden_dim, embedding_dim, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        self.conv_first = SimpleGCNLayer(input_dim, hidden_dim)
        self.conv_block = nn.ModuleList([
            SimpleGCNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers - 2)
        ])
        self.conv_last = SimpleGCNLayer(hidden_dim, embedding_dim)
        self.num_layers = num_layers

        # JK concatenation of all layers as input to classifier
        pred_dim = hidden_dim * (num_layers - 1) + embedding_dim
        self.cls = nn.Linear(pred_dim, num_classes)

    def forward(self, x, adj):
        """
        Args:
            x: [B, N, D] or [N, D]
            adj: [B, N, N] or [N, N]
        Returns:
            logits: [N, num_classes]
        """
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            added_batch = True

        h = self.conv_first(x, adj)
        h = F.relu(h)
        h = self.dropout(h)
        feats = [h]

        for layer in self.conv_block:
            h = layer(h, adj)
            h = F.relu(h)
            h = self.dropout(h)
            feats.append(h)

        h = self.conv_last(h, adj)
        feats.append(h)

        # Concatenate all layer outputs (JK-style)
        out = torch.cat(feats, dim=-1)  # [B, N, sum_dim]
        logits = self.cls(out)          # [B, N, num_classes]

        if added_batch:
            logits = logits.squeeze(0)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# GLEM Components
# ──────────────────────────────────────────────────────────────────────────────

class GLEMGNN(nn.Module):
    """GLEM's GNN: takes node features + (optional) LM pseudo-labels."""

    def __init__(self, input_dim, num_classes, hidden_dim=64, embedding_dim=64,
                 num_layers=3, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        # input_dim = original feature dim + num_classes (for LM pseudo-label concatenation)
        self.gnn = SimpleGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x, adj, lm_pseudo=None):
        """
        Args:
            x: [N, D] or [B, N, D] node features
            adj: [N, N] or [B, N, N] adjacency
            lm_pseudo: [N, num_classes] or [B, N, num_classes] or None
        Returns:
            logits: [N, num_classes]
        """
        # If no pseudo-labels provided, use uniform distribution as placeholder
        # (following GLEM paper: inference always requires LM-generated pseudo-labels)
        if lm_pseudo is None:
            uniform_val = 1.0 / self.num_classes
            if x.dim() == 3:
                lm_pseudo = torch.full(
                    (x.size(0), x.size(1), self.num_classes),
                    uniform_val, device=x.device, dtype=x.dtype
                )
            else:
                lm_pseudo = torch.full(
                    (x.size(0), self.num_classes),
                    uniform_val, device=x.device, dtype=x.dtype
                )

        # Make x and lm_pseudo have matching dimensions
        if lm_pseudo.dim() != x.dim():
            if x.dim() - lm_pseudo.dim() == 1:
                lm_pseudo = lm_pseudo.unsqueeze(0)
            elif lm_pseudo.dim() - x.dim() == 1:
                x = x.unsqueeze(0)
                adj = adj.unsqueeze(0)

        if lm_pseudo is not None:
            x = torch.cat([x, lm_pseudo], dim=-1)
        return self.gnn(x, adj)


class GLEMLM(nn.Module):
    """GLEM's LM: MLP over text embeddings."""

    def __init__(self, text_dim, hidden_dim, num_classes, text_encoder=None, dropout=0.3):
        super().__init__()
        self.text_encoder = text_encoder
        if text_encoder is not None:
            text_dim = text_encoder.fc.out_features
        self.text_dim = text_dim
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.celoss = nn.CrossEntropyLoss()

    def encode_text(self, texts):
        """Encode texts via the project's TextEncoder (with fallback)."""
        if self.text_encoder is not None:
            return self.text_encoder(texts)
        # Should never happen since we always pass a text encoder
        return torch.randn(len(texts), self.text_dim)

    def forward(self, text_feats):
        if isinstance(text_feats, (list, tuple)) and len(text_feats) > 0 and isinstance(text_feats[0], str):
            feats = self.encode_text(text_feats)
        else:
            feats = text_feats
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float)
        return self.mlp(feats)

    def loss(self, pred, label, mask=None):
        if mask is not None:
            return self.celoss(pred[mask], label[mask])
        return self.celoss(pred, label)


# ──────────────────────────────────────────────────────────────────────────────
# GLEM EM Framework
# ──────────────────────────────────────────────────────────────────────────────

class GLEMFramework:
    """GLEM's EM iterative training framework.

    Alternates:
      - E-step: Update pseudo-labels from current GNN and LM
      - M-step: Train GNN with LM pseudo-labels; train LM with GNN pseudo-labels
    """

    def __init__(self, gnn, lm, num_em_iters=3, inner_epochs=50,
                 confidence_threshold=0.5, device="cpu"):
        self.gnn = gnn.to(device)
        self.lm = lm.to(device)
        self.num_em_iters = num_em_iters
        self.inner_epochs = inner_epochs
        self.confidence_threshold = confidence_threshold
        self.device = device

    def _filter_by_confidence(self, probs):
        """Keep only confident predictions; zero out uncertain ones."""
        max_probs, _ = probs.max(dim=-1)
        confident = max_probs >= self.confidence_threshold
        pseudo = probs.clone()
        pseudo[~confident] = 0.0
        return pseudo

    def train_em(self, x, adj, labels, texts, train_idx, val_idx, test_idx):
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)
        train_mask = torch.zeros(len(labels), dtype=torch.bool, device=self.device)
        train_mask[train_idx] = True

        num_classes = self.gnn.num_classes
        gt_soft = F.one_hot(labels_t, num_classes=num_classes).float()
        uniform = torch.ones_like(gt_soft) / num_classes
        init_pseudo = torch.where(train_mask.unsqueeze(-1), gt_soft, uniform)

        lm_pseudo = init_pseudo.clone()
        gnn_pseudo = init_pseudo.clone()

        best_val_acc = 0.0
        best_test_acc = 0.0

        for em_iter in range(self.num_em_iters):
            print(f"    EM iter {em_iter+1}/{self.num_em_iters}")

            # M-step: train GNN with LM pseudo-labels
            self._train_gnn(x, adj, lm_pseudo, train_mask, labels_t, epochs=self.inner_epochs)

            # M-step: train LM with GNN pseudo-labels
            self._train_lm(texts, gnn_pseudo, train_mask, labels_t, epochs=self.inner_epochs)

            # E-step: regenerate pseudo-labels
            self.gnn.eval()
            with torch.no_grad():
                gnn_logits = self.gnn(x.to(self.device), adj.to(self.device))
                gnn_probs = F.softmax(gnn_logits, dim=-1)

            self.lm.eval()
            with torch.no_grad():
                lm_logits = self.lm(texts)
                lm_probs = F.softmax(lm_logits, dim=-1)

            gnn_filtered = self._filter_by_confidence(gnn_probs)
            lm_filtered = self._filter_by_confidence(lm_probs)
            gnn_pseudo = torch.where(train_mask.unsqueeze(-1), gt_soft, gnn_filtered)
            lm_pseudo = torch.where(train_mask.unsqueeze(-1), gt_soft, lm_filtered)

            # Evaluate
            self.gnn.eval()
            with torch.no_grad():
                gnn_logits = self.gnn(x.to(self.device), adj.to(self.device))
                if gnn_logits.dim() == 3:
                    gnn_logits = gnn_logits.squeeze(0)
                gnn_preds = gnn_logits.argmax(dim=-1).cpu().numpy()
                train_acc = (gnn_preds[train_idx] == labels[train_idx]).mean()
                val_acc = (gnn_preds[val_idx] == labels[val_idx]).mean()
                test_acc = (gnn_preds[test_idx] == labels[test_idx]).mean()

            self.lm.eval()
            with torch.no_grad():
                lm_preds = self.lm(texts).argmax(dim=-1).cpu().numpy()
                lm_train_acc = (lm_preds[train_idx] == labels[train_idx]).mean()
                lm_val_acc = (lm_preds[val_idx] == labels[val_idx]).mean()
                lm_test_acc = (lm_preds[test_idx] == labels[test_idx]).mean()

            print(f"      GNN  train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")
            print(f"      LM   train={lm_train_acc:.4f} val={lm_val_acc:.4f} test={lm_test_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

        return {"best_val_acc": best_val_acc, "best_test_acc": best_test_acc}

    def _train_gnn(self, x, adj, lm_pseudo, train_mask, labels_t, epochs=50):
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.01, weight_decay=5e-4)
        self.gnn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.gnn(x.to(self.device), adj.to(self.device), lm_pseudo=lm_pseudo)
            # logits may be [1, N, C] or [N, C] depending on input dim
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            ce_loss = F.cross_entropy(logits[train_mask], labels_t[train_mask])
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, lm_pseudo, reduction="batchmean")
            loss = ce_loss + 0.5 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 5.0)
            optimizer.step()

    def _train_lm(self, texts, gnn_pseudo, train_mask, labels_t, epochs=50):
        optimizer = torch.optim.Adam(self.lm.parameters(), lr=0.01, weight_decay=5e-4)
        self.lm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.lm(texts)
            ce_loss = F.cross_entropy(logits[train_mask], labels_t[train_mask])
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, gnn_pseudo, reduction="batchmean")
            loss = ce_loss + 0.5 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lm.parameters(), 5.0)
            optimizer.step()
