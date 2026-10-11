"""
Baselines package for multimodal graph neural networks.

This package contains implementations of popular multimodal GNN baselines:
- MAGNN: Metapath Aggregated Graph Neural Network (WWW 2019)
- GATE: Gated Multimodal Networks

Usage:
    from baselines.magnn import MAGNNEncoder
    from baselines.gate import GATEModel
"""

from .magnn import MAGNNEncoder, MAGNNNodeEncoder
from .gate import GATEModel, GATENodeEncoder

__all__ = [
    'MAGNNEncoder',
    'MAGNNNodeEncoder',
    'GATEModel',
    'GATENodeEncoder',
]

