"""
MAGNN: Metapath Aggregated Graph Neural Network
Reference: "MAGNN: Metapath Aggregated Graph Neural Network" (WWW 2019)
https://arxiv.org/abs/1905.09550

This is an adapted version for text-attributed graphs.
Core idea: Use metapath-based aggregation with attention mechanism to
combine structural and textual information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GraphConv, GcnEncoderGraph
from multimodal.text_encoder import TextEncoder


class MAGNNEncoder(GcnEncoderGraph):
    """
    MAGNN-style encoder for text-attributed graphs.

    Key components:
    1. Metapath-based aggregation (simplified: direct neighbors)
    2. Modality encoder (graph + text)
    3. Semantic-level attention fusion
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 text_dim=128, fusion_method='magnn', **kwargs):
        force_fallback = kwargs.pop('force_text_fallback', False)
        disable_text = kwargs.pop('disable_text', False)
        pred_hidden_dims = kwargs.get('pred_hidden_dims', [])
        super().__init__(input_dim, hidden_dim, embedding_dim, label_dim, num_layers, **kwargs)

        # Store dimensions
        self.text_dim = text_dim
        self.disable_text = disable_text
        self.graph_feature_dim = self.pred_input_dim

        # Text encoder
        self.text_encoder = TextEncoder(output_dim=text_dim, force_fallback=force_fallback)

        # MAGNN-style semantic attention
        # Projects both modalities to a common semantic space
        self.semantic_proj = nn.Linear(hidden_dim + text_dim, hidden_dim)
        self.semantic_attention = nn.Parameter(torch.FloatTensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.semantic_attention.data)

        # Modality-specific transformations
        self.graph_transform = nn.Linear(self.graph_feature_dim, hidden_dim)
        self.text_transform = nn.Linear(text_dim, hidden_dim)

        # Final prediction input dimension
        self.pred_input_dim = hidden_dim

        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim
        )

    def forward(self, x, adj, text="", batch_num_nodes=None, **kwargs):
        """
        Forward pass for MAGNN.

        Args:
            x: Node features [batch, num_nodes, input_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            text: Text features [batch, text_dim] or list of strings
        """
        # 1. Graph structural encoding
        graph_feat, adj_att = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last)

        # Handle concatenated features
        if hasattr(self, 'concat') and not self.concat:
            last_dim = self.conv_last.output_dim
            graph_feat = graph_feat[:, :, -last_dim:]

        # Graph-level pooling
        graph_feat_pooled = torch.max(graph_feat, dim=1)[0]  # [batch, graph_dim]
        graph_feat_hidden = self.graph_transform(graph_feat_pooled)  # [batch, hidden_dim]

        # 2. Text encoding (if available)
        if not self.disable_text:
            if isinstance(text, str):
                text_feat = self.text_encoder.encode_text([text])  # [1, text_dim]
            elif isinstance(text, torch.Tensor):
                text_feat = text
            else:
                text_feat = self.text_encoder.encode_text(text)

            # Ensure text_feat has batch dimension
            if text_feat.dim() == 2 and text_feat.size(0) != graph_feat.size(0):
                text_feat = text_feat.expand(graph_feat.size(0), -1)

            text_feat_hidden = self.text_transform(text_feat)  # [batch, hidden_dim]
        else:
            text_feat_hidden = torch.zeros_like(graph_feat_hidden)

        # 3. MAGNN-style semantic fusion with attention
        # Concatenate and project to semantic space
        combined = torch.cat([graph_feat_hidden, text_feat_hidden], dim=-1)
        semantic = torch.tanh(self.semantic_proj(combined))

        # Compute attention weights
        att_weights = torch.matmul(semantic, self.semantic_attention)  # [batch, 1]
        att_weights = F.softmax(att_weights, dim=0)

        # Weighted fusion
        fused_feat = att_weights * graph_feat_hidden + (1 - att_weights) * text_feat_hidden

        # 4. Prediction
        pred = self.pred_model(fused_feat)
        return pred


class MAGNNNodeEncoder(nn.Module):
    """
    Node-level MAGNN for graph-level prediction.
    Aggregates both structural and textual node features.
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

        # MAGNN inter-modal attention
        self.graph_proj = nn.Linear(self.graph_encoder.pred_input_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion_attention = nn.Linear(hidden_dim * 2, 1)

        # Output dimension
        self.pred_input_dim = hidden_dim
        self.pred_model = nn.Linear(self.pred_input_dim, label_dim)

    def forward(self, x, adj, text=None, batch_num_nodes=None, **kwargs):
        # Graph encoding
        graph_feat, adj_att = self.graph_encoder.gcn_forward(
            x, adj,
            self.graph_encoder.conv_first,
            self.graph_encoder.conv_block,
            self.graph_encoder.conv_last
        )

        # Node-level pooling
        graph_pooled = torch.max(graph_feat, dim=1)[0]
        graph_hidden = torch.relu(self.graph_proj(graph_pooled))

        # Text encoding
        if text is not None:
            if isinstance(text, list):
                text_feat = self.text_encoder.encode_text(text)
            else:
                text_feat = text
            text_hidden = torch.relu(self.text_proj(text_feat))
        else:
            text_hidden = torch.zeros_like(graph_hidden)

        # MAGNN-style attention fusion
        combined = torch.cat([graph_hidden, text_hidden], dim=-1)
        att_score = self.fusion_attention(combined)
        att_weights = F.softmax(att_score, dim=0)

        fused = att_weights * graph_hidden + (1 - att_weights) * text_hidden

        # Prediction
        pred = self.pred_model(fused)
        return pred, adj_att

