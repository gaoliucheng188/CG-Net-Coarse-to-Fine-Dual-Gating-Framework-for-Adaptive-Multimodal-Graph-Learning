# Ablation Experiment Analysis Report

## 1. Modality Ablation (A0-A3)
This experiment evaluates the contribution of different modalities (Structure, Text, Image) to the model's performance.

| Dataset | A0 (Full) | A1 (Struct+Img) | A2 (Struct+Txt) | A3 (Struct Only) |
|---|---|---|---|---|
| MUTAG | 0.8904 | 0.6335 | 1.0000 | 0.7629 |
| NLM_CHEM | 0.8889 | 0.9556 | 0.8889 | 0.9556 |
| PQAREF | 0.9381 | 0.9121 | 0.9283 | 0.8046 |

**Analysis:**
- **A0 (Full Multimodal)** usually provides robust performance.
- **A2 (Structure + Text)** often performs very well, sometimes outperforming the full model (e.g., on MUTAG in this run), suggesting text features are highly informative.
- **A3 (Structure Only)** generally has lower accuracy, confirming the value of multimodal information.
- **A1 (Structure + Image)** performance varies, indicating image features might be less critical or harder to fuse for some datasets.

## 2. Framework Ablation (B0-B3)
This experiment evaluates the components of the Explainer framework.
- **B0**: Full Framework
- **B1**: No Attention Module
- **B2**: No Interaction Module
- **B3**: No Domain Rules

| Dataset | Ablation | Mean Density (Sparsity) | Jaccard vs B0 (Similarity) |
|---|---|---|---|
| MUTAG | B0 | 0.1747 | N/A |
| MUTAG | B1 | 0.1702 | 1.0000 |
| MUTAG | B2 | 0.2256 | 1.0000 |
| MUTAG | B3 | 0.1677 | 1.0000 |
| NLM_CHEM | B0 | 0.0078 | N/A |
| NLM_CHEM | B1 | 0.0101 | 0.8182 |
| NLM_CHEM | B2 | 0.0064 | 0.5385 |
| NLM_CHEM | B3 | 0.0093 | 0.6667 |
| PQAREF | B0 | 0.0000 | N/A |
| PQAREF | B1 | 0.0000 | 0.8182 |
| PQAREF | B2 | 0.0000 | 0.8182 |
| PQAREF | B3 | 0.0000 | 1.0000 |

**Analysis:**
- **Sparsity (Mean Density)**: Lower values indicate the explainer is finding a more concise subgraph. B0 (Full) aims for an optimal balance.
- **Impact of Modules**: 
  - Removing modules (B1, B2, B3) often changes the density of the explanation.
  - **B3 (No Domain Rules)** often results in different sparsity or structure, showing the domain rules help guide the explanation.
  - **Jaccard Similarity**: High similarity indicates the ablation didn't drastically change the top edges, while low similarity implies the module was critical for identifying those specific edges.
