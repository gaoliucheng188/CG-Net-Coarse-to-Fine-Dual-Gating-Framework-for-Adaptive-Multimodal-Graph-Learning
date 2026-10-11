"""
GATE: Gated Multimodal Networks
Reference: "Gated Multimodal Networks" (arXiv:1905.09690)

This implements a simple but effective gating mechanism for multimodal fusion
in text-attributed graphs. The key idea is to use a gating signal to control
how much each modality contributes to the final prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GraphConv, GcnEncoderGraph
from multimodal.text_encoder import TextEncoder


class GATEModel(nn.Module):
    """
    Gated Multimodal Network for text-attributed graphs.

    Key components:
    1. Separate encoders for each modality (graph, text)
    2. A gating mechanism that learns to weight modality contributions
    3. Fused representation for prediction
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 text_dim=128, fusion_method='gate', gate_hidden_dim=64, **kwargs):
        super().__init__()
        force_fallback = kwargs.pop('force_text_fallback', False)
        disable_text = kwargs.pop('disable_text', False)
        pred_hidden_dims = kwargs.get('pred_hidden_dims', [])

        # Graph encoder (same as base GCN)
        self.graph_encoder = GcnEncoderGraph(
            input_dim, hidden_dim, embedding_dim, label_dim, num_layers, **kwargs
        )

        # Text encoder
        self.text_encoder = TextEncoder(output_dim=text_dim, force_fallback=force_fallback)

        # Modality embeddings projection
        self.graph_dim = self.graph_encoder.pred_input_dim
        self.text_dim = text_dim

        # Project each modality to common dimension
        self.graph_proj = nn.Linear(self.graph_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Gating network: takes concatenated modality features and outputs gate value
        # Based on Gated Multimodal Networks paper
        self.gate_fc1 = nn.Linear(hidden_dim * 2, gate_hidden_dim)
        self.gate_fc2 = nn.Linear(gate_hidden_dim, 1)

        # Output projection
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Final prediction layers
        self.pred_input_dim = hidden_dim
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim
        )

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim):
        if len(pred_hidden_dims) == 0:
            return nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(nn.ReLU())
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            return nn.Sequential(*pred_layers)

    def forward(self, x, adj, text="", batch_num_nodes=None, **kwargs):
        """
        Forward pass with gating mechanism.

        Args:
            x: Node features [batch, num_nodes, input_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            text: Text features [batch, text_dim] or string
        """
        # 1. Graph encoding
        graph_feat, adj_att = self.graph_encoder.gcn_forward(
            x, adj,
            self.graph_encoder.conv_first,
            self.graph_encoder.conv_block,
            self.graph_encoder.conv_last
        )

        # Handle concatenated features
        if hasattr(self.graph_encoder, 'concat') and not self.graph_encoder.concat:
            last_dim = self.graph_encoder.conv_last.output_dim
            graph_feat = graph_feat[:, :, -last_dim:]

        # Graph-level pooling
        graph_pooled = torch.max(graph_feat, dim=1)[0]  # [batch, graph_dim]
        graph_hidden = torch.relu(self.graph_proj(graph_pooled))  # [batch, hidden_dim]

        # 2. Text encoding
        if isinstance(text, str):
            text_feat = self.text_encoder.encode_text([text])  # [1, text_dim]
        elif isinstance(text, torch.Tensor):
            text_feat = text
        else:
            text_feat = self.text_encoder.encode_text(text)

        # Ensure batch dimension matches
        if text_feat.size(0) != graph_feat.size(0):
            text_feat = text_feat.expand(graph_feat.size(0), -1)

        text_hidden = torch.relu(self.text_proj(text_feat))  # [batch, hidden_dim]

        # 3. Gating mechanism (GATE core)
        # Concatenate modalities for gate computation
        concat_hidden = torch.cat([graph_hidden, text_hidden], dim=-1)  # [batch, hidden*2]

        # Gate computation: g = sigmoid(W2 * tanh(W1 * concat))
        gate_input = torch.tanh(self.gate_fc1(concat_hidden))
        gate_value = torch.sigmoid(self.gate_fc2(gate_input))  # [batch, 1]

        # 4. Gated fusion
        # z = g * h_graph + (1-g) * h_text
        fused_hidden = gate_value * graph_hidden + (1 - gate_value) * text_hidden

        # Additional projection after fusion
        fused_proj = torch.relu(self.fusion_proj(concat_hidden))

        # 5. Prediction
        pred = self.pred_model(fused_proj)
        return pred


class GATENodeEncoder(nn.Module):
    """
    Node-level GATE for graph-level prediction.
    Uses gating at both node and graph level.
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 text_dim=128, **kwargs):
        super().__init__()
        force_fallback = kwargs.pop('force_text_fallback', False)

        # Graph encoder
        self.graph_encoder = GcnEncoderGraph(
            input_dim, hidden_dim, embedding_dim, label_dim, num_layers, **kwargs
        )

        # Text encoder
        self.text_encoder = TextEncoder(output_dim=text_dim, force_fallback=force_fallback)

        # Projections
        self.graph_proj = nn.Linear(self.graph_encoder.pred_input_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Node-level gate
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Graph-level gate
        self.graph_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Output
        self.pred_model = nn.Linear(hidden_dim, label_dim)

    def forward(self, x, adj, text=None, batch_num_nodes=None, **kwargs):
        # Graph encoding
        graph_feat, adj_att = self.graph_encoder.gcn_forward(
            x, adj,
            self.graph_encoder.conv_first,
            self.graph_encoder.conv_block,
            self.graph_encoder.conv_last
        )

        # Text encoding
        if text is not None:
            if isinstance(text, list):
                text_feat = self.text_encoder.encode_text(text)
            else:
                text_feat = text
            text_hidden = torch.relu(self.text_proj(text_feat))
        else:
            text_hidden = torch.zeros(graph_feat.size(0), self.text_encoder.fc.out_features,
                                      device=graph_feat.device)

        # Node-level gating (optional refinement)
        graph_hidden = torch.relu(self.graph_proj(graph_feat))

        # Match dimensions for node-level text
        if text_hidden.size(-1) != graph_hidden.size(-1):
            text_proj_node = nn.Linear(text_hidden.size(-1), graph_hidden.size(-1)).to(graph_hidden.device)
            text_hidden = text_proj_node(text_hidden)

        if text_hidden.dim() == 2:
            text_hidden = text_hidden.unsqueeze(1).expand(-1, graph_hidden.size(1), -1)

        # Node-level gated fusion
        concat_node = torch.cat([graph_hidden, text_hidden], dim=-1)
        node_gate = self.node_gate(concat_node)
        fused_node = node_gate * graph_hidden + (1 - node_gate) * text_hidden

        # Graph-level pooling
        fused_graph = torch.max(fused_node, dim=1)[0]

        # Graph-level gating
        graph_hidden_pooled = torch.max(graph_hidden, dim=1)[0]
        if text_hidden.size(0) == 1 and graph_feat.size(0) > 1:
            text_hidden_pooled = text_hidden.mean(dim=1)
        else:
            text_hidden_pooled = text_hidden.mean(dim=1) if text_hidden.dim() == 3 else text_hidden

        concat_graph = torch.cat([graph_hidden_pooled, text_hidden_pooled], dim=-1)
        graph_gate = self.graph_gate(concat_graph)

        # Final gated fusion
        final_fused = graph_gate * graph_hidden_pooled + (1 - graph_gate) * text_hidden_pooled

        # Prediction
        pred = self.pred_model(final_fused)
        return pred, adj_att

