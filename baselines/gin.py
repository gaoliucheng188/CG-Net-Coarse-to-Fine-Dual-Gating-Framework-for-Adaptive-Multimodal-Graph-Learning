"""
GIN: Graph Isomorphism Network
Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
https://arxiv.org/abs/1810.00826

Pure PyTorch implementation (no PyG dependency) that plugs into the
existing CG-Net data pipeline (networkx + dense adjacency).

GIN updates:
    h_v^{(k)} = MLP^{(k)}((1 + eps^{(k)}) * h_v^{(k-1)} + SUM_{u in N(v)} h_u^{(k-1)})

For graph classification: readout = SUM_{k} GlobalMeanPool(h^{(k)})
For node classification:  use final-layer node embeddings directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GINConv(nn.Module):
    """Single GIN convolution layer (dense adjacency version).

    h' = MLP((1 + eps) * h + adj @ h)
    """

    def __init__(self, input_dim, output_dim, eps_learnable=True):
        super().__init__()
        self.eps = nn.Parameter(torch.Tensor(1)) if eps_learnable else 0.0
        if eps_learnable:
            init.constant_(self.eps, 0.0)

        # Two-layer MLP as in the original paper
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x, adj):
        """
        Args:
            x:   [batch, num_nodes, input_dim]
            adj: [batch, num_nodes, num_nodes]  (dense, unnormalized)
        Returns:
            out: [batch, num_nodes, output_dim]
        """
        # Aggregate neighbours: adj @ x  ->  [batch, num_nodes, input_dim]
        agg = torch.matmul(adj, x)

        # (1 + eps) * self + aggregate
        if isinstance(self.eps, nn.Parameter):
            out = (1.0 + self.eps) * x + agg
        else:
            out = (1.0 + self.eps) * x + agg

        # MLP: need to reshape for BatchNorm1d which expects [N, C]
        batch_size, num_nodes, feat_dim = out.shape
        out_2d = out.reshape(batch_size * num_nodes, feat_dim)
        out_2d = self.mlp(out_2d)
        out = out_2d.reshape(batch_size, num_nodes, -1)

        return out


class GINNodeEncoder(nn.Module):
    """GIN for node-level classification (same task as CG-Net's GcnEncoderNode).

    Architecture mirrors the project's GcnEncoderNode:
      - 3 GIN layers with hidden_dim -> hidden_dim -> embedding_dim
      - Concatenate all layer outputs (JK-style) for final prediction
      - Per-node linear classifier
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers=3,
        concat=True,
        bn=True,
        dropout=0.5,
        eps_learnable=True,
        args=None,
    ):
        super().__init__()
        self.concat = concat
        self.num_layers = num_layers
        self.label_dim = label_dim
        self.bn = bn

        # Build GIN layers
        self.conv_first = GINConv(input_dim, hidden_dim, eps_learnable)
        self.conv_block = nn.ModuleList(
            [GINConv(hidden_dim, hidden_dim, eps_learnable) for _ in range(num_layers - 2)]
        )
        self.conv_last = GINConv(hidden_dim, embedding_dim, eps_learnable)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Prediction head
        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = nn.Linear(self.pred_input_dim, label_dim)

        # Loss
        self.celoss = nn.CrossEntropyLoss()

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def gcn_forward(self, x, adj):
        """Forward through all GIN layers, return concatenated node embeddings.

        Returns:
            x_tensor: [batch, num_nodes, pred_input_dim]
        """
        x = self.conv_first(x, adj)
        x = self.act(x)
        x = self.dropout(x)
        x_all = [x]

        for conv in self.conv_block:
            x = conv(x, adj)
            x = self.act(x)
            x = self.dropout(x)
            x_all.append(x)

        x = self.conv_last(x, adj)
        x_all.append(x)

        # Concatenate all layer outputs along feature dim
        x_tensor = torch.cat(x_all, dim=2)  # [batch, num_nodes, pred_input_dim]
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        """
        Args:
            x:   [batch, num_nodes, input_dim]
            adj: [batch, num_nodes, num_nodes]
        Returns:
            pred:     [batch, num_nodes, label_dim]
            adj_att:  None (GIN has no attention)
        """
        embedding = self.gcn_forward(x, adj)  # [batch, N, pred_input_dim]
        pred = self.pred_model(embedding)      # [batch, N, label_dim]
        return pred, None

    def loss(self, pred, label):
        """Cross-entropy loss, compatible with GcnEncoderNode.loss()."""
        if pred.dim() == 3:
            pred = pred.permute(0, 2, 1)  # [batch, C, N]
            return self.celoss(pred, label)
        elif pred.dim() == 2:
            return self.celoss(pred, label)
        else:
            raise ValueError(f"Unexpected pred dimension in loss: {pred.dim()}")


class GINGraphEncoder(nn.Module):
    """GIN for graph-level classification.

    Uses global mean pooling + sum of all layer representations (JK-style).
    Compatible with the project's GcnEncoderGraph interface.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers=3,
        concat=True,
        bn=True,
        dropout=0.5,
        eps_learnable=True,
        args=None,
    ):
        super().__init__()
        self.concat = concat
        self.num_layers = num_layers
        self.label_dim = label_dim

        self.conv_first = GINConv(input_dim, hidden_dim, eps_learnable)
        self.conv_block = nn.ModuleList(
            [GINConv(hidden_dim, hidden_dim, eps_learnable) for _ in range(num_layers - 2)]
        )
        self.conv_last = GINConv(hidden_dim, embedding_dim, eps_learnable)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = nn.Linear(self.pred_input_dim, label_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        """
        Args:
            x:   [batch, num_nodes, input_dim]
            adj: [batch, num_nodes, num_nodes]
        Returns:
            ypred:    [batch, label_dim]
            adj_att:  None
        """
        x = self.conv_first(x, adj)
        x = self.act(x)
        x = self.dropout(x)
        out_all = [torch.mean(x, dim=1)]  # global mean pool

        for conv in self.conv_block:
            x = conv(x, adj)
            x = self.act(x)
            x = self.dropout(x)
            out_all.append(torch.mean(x, dim=1))

        x = self.conv_last(x, adj)
        out_all.append(torch.mean(x, dim=1))

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out_all[-1]

        ypred = self.pred_model(output)
        return ypred, None

    def loss(self, pred, label, type="softmax"):
        if type == "softmax":
            return F.cross_entropy(pred, label, size_average=True)
        elif type == "margin":
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim, device=pred.device).long()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

