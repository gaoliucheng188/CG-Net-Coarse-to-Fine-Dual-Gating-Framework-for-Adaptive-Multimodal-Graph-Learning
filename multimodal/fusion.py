import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, graph_dim, text_dim, image_dim, output_dim, fusion_method='concat', gpu=True, disable_text=False, disable_image=False, project_concat=True):
        super().__init__()
        self.fusion_method = fusion_method
        self.gpu = gpu
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.disable_text = disable_text
        self.disable_image = disable_image
        self.project_concat = project_concat
        self.output_dim = output_dim

        if fusion_method == 'concat':
            input_dim = graph_dim
            if not disable_text:
                input_dim += text_dim
            if not disable_image:
                input_dim += image_dim
            
            if project_concat:
                self.fc = nn.Linear(input_dim, output_dim)
            else:
                self.fc = None
        elif fusion_method == 'bilinear':
            self.graph_proj = nn.Linear(graph_dim, output_dim)
            if not disable_text:
                self.text_proj = nn.Linear(text_dim, output_dim)
            if not disable_image:
                self.image_proj = nn.Linear(image_dim, output_dim)
            self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
            self.fc = nn.Linear(output_dim, output_dim)
        elif fusion_method == 'attention':
            self.query_proj = nn.Linear(graph_dim, graph_dim)
            if not disable_text:
                self.text_proj = nn.Linear(text_dim, graph_dim)
            if not disable_image:
                self.image_proj = nn.Linear(image_dim, graph_dim)
            self.scale = graph_dim ** -0.5 if graph_dim > 0 else 1.0
            self.fc = nn.Linear(graph_dim, output_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        # Do not move to CUDA here; caller should move module to device when ready.

    def forward(self, graph_feat, text_feat, image_feat):
        # 检查哪些模态可用
        has_text = not self.disable_text and text_feat is not None
        has_image = not self.disable_image and image_feat is not None
        
        # Only filter out zero-tensors if we are projecting (dynamic input size supported)
        # If project_concat is False, we must preserve dimensions matching the checkpoint
        if not (self.fusion_method == 'concat' and not self.project_concat):
            if has_text:
                has_text = not torch.all(text_feat == 0)
            if has_image:
                has_image = not torch.all(image_feat == 0)
        
        if self.fusion_method == 'concat':
            features = [graph_feat]
            if has_text:
                features.append(text_feat)
            if has_image:
                features.append(image_feat)
            
            fused = torch.cat(features, dim=-1)
            
            if self.fc is not None:
                # 动态调整输入维度
                current_dim = fused.shape[-1]
                if current_dim != self.fc.in_features:
                    # 创建新的线性层
                    new_fc = nn.Linear(current_dim, self.fc.out_features).to(fused.device)
                    # 复制权重，如果维度匹配
                    min_dim = min(current_dim, self.fc.in_features)
                    new_fc.weight.data[:, :min_dim] = self.fc.weight.data[:, :min_dim]
                    new_fc.bias.data = self.fc.bias.data
                    self.fc = new_fc
                
                fused = self.fc(fused)
        elif self.fusion_method == 'bilinear':
            g = self.graph_proj(graph_feat)
            if has_text:
                t = self.text_proj(text_feat)
                fused = self.bilinear(g, t)
            else:
                fused = g
            if has_image:
                i = self.image_proj(image_feat)
                fused = fused + i
            fused = self.fc(fused)
        elif self.fusion_method == 'attention':
            # Equation 2: alpha = Softmax(W_attn * tanh(W_proj * Concat(h_G, h_T, h_I)))
            # We implement this "Additive Attention" mechanism.
            
            # 1. Project each modality to a common dimension (graph_dim) if needed
            # Note: In the equation, W_proj acts on the concatenation. 
            # However, usually dimensions differ, so we project first or concat then project.
            # If we concat first, the input dim is sum of dims.
            
            features = []
            # Graph feature is always present
            # We use graph_feat as the reference for shape (e.g. [B, N, D] or [B, D])
            features.append(graph_feat)
            
            # Helper to create zero tensor matching graph_feat shape but with different feature dim
            def get_zero_tensor(feat_dim):
                shape = list(graph_feat.shape)
                shape[-1] = feat_dim
                return torch.zeros(shape, device=graph_feat.device)

            if has_text:
                # Ensure dimensions match graph_feat (e.g. if graph is 3D and text is 2D, unsqueeze?)
                # But usually they should match from the encoder.
                # Just in case _ensure_2d was used before or something.
                if text_feat.dim() < graph_feat.dim():
                     # Assuming missing node dim
                     text_feat = text_feat.unsqueeze(1).expand(-1, graph_feat.shape[1], -1)
                features.append(text_feat)
            else:
                # If missing but we need to maintain dimension for W_proj?
                # Usually we just ignore it or use zero padding.
                # If we use zero padding, we need to know the dimension.
                if not self.disable_text:
                     features.append(get_zero_tensor(self.text_dim))

            if has_image:
                if image_feat.dim() < graph_feat.dim():
                     image_feat = image_feat.unsqueeze(1).expand(-1, graph_feat.shape[1], -1)
                features.append(image_feat)
            else:
                if not self.disable_image:
                     features.append(get_zero_tensor(self.image_dim))

            # Concatenate: [Batch, ..., Sum_Dims]
            concatenated = torch.cat(features, dim=-1)
            
            # Initialize weights if not done (lazy init based on input shape)
            if not hasattr(self, 'attn_proj'):
                input_dim = concatenated.shape[-1]
                # W_proj
                self.attn_proj = nn.Linear(input_dim, self.graph_dim).to(concatenated.device)
                # W_attn (projects to scalar score per modality? No, usually attention over items)
                # Here we have 3 items (modalities). 
                # But the equation suggests a single alpha vector? 
                # "alpha = Softmax(...)"
                # If alpha is a vector of weights for h_G, h_T, h_I, then we need to compute 3 scores.
                # But we concatenated them.
                # Alternative interpretation: The equation computes a weight vector for the *fused* representation?
                # No, usually attention weights sum to 1.
                
                # Let's implement: 
                # 1. Project each modality to common hidden dim D.
                # 2. Stack them: [B, 3, D]
                # 3. Compute attention scores for each of the 3 vectors.
                #    score_i = v^T tanh(W h_i)
                
                # BUT the user's equation is: alpha = Softmax(W_attn * tanh(W_proj * Concat(h_G, h_T, h_I)))
                # This looks like self-attention or a gating mechanism where the weights depend on the joint state.
                # Let's assume alpha has dimension 3 (or number of modalities).
                
                num_modalities = 1 + (1 if not self.disable_text else 0) + (1 if not self.disable_image else 0)
                self.attn_score = nn.Linear(self.graph_dim, num_modalities).to(concatenated.device)
                
                # We also need projections to map original features to output_dim for the final sum
                self.modality_projs = nn.ModuleList()
                # Graph
                self.modality_projs.append(nn.Linear(self.graph_dim, self.output_dim).to(concatenated.device))
                # Text
                if not self.disable_text:
                    self.modality_projs.append(nn.Linear(self.text_dim, self.output_dim).to(concatenated.device))
                # Image
                if not self.disable_image:
                    self.modality_projs.append(nn.Linear(self.image_dim, self.output_dim).to(concatenated.device))

            # Calculate Alpha
            # h_joint = Tanh(W_proj * Concat)
            h_joint = torch.tanh(self.attn_proj(concatenated))
            # scores = W_attn * h_joint
            scores = self.attn_score(h_joint) # [B, ..., Num_Modalities]
            alpha = torch.softmax(scores, dim=-1) # [B, ..., Num_Modalities]
            
            # Weighted Sum
            # We need to project individual features to output_dim first
            weighted_sum = 0
            idx = 0
            
            # Graph
            h_G_proj = self.modality_projs[0](graph_feat)
            weighted_sum += alpha[..., 0:1] * h_G_proj
            idx += 1
            
            # Text
            if not self.disable_text:
                if has_text:
                    h_T_proj = self.modality_projs[idx](text_feat)
                    weighted_sum += alpha[..., idx:idx+1] * h_T_proj
                # If missing text, weight is 0 effectively (or we ignore it)
                # But alpha is computed based on zero-padded input.
                idx += 1
                
            # Image
            if not self.disable_image:
                if has_image:
                    h_I_proj = self.modality_projs[idx](image_feat)
                    weighted_sum += alpha[..., idx:idx+1] * h_I_proj
                idx += 1
                
            fused = weighted_sum
            
            # Final FC (optional, as we already projected to output_dim)
            # But to match other methods structure, we can pass through FC or just return.
            # The other methods have self.fc at the end.
            # Let's assume self.fc is Identity or further processing.
            # If we used output_dim in projections, we might not need self.fc, 
            # but the constructor initializes self.fc.
            # Let's use self.fc if it exists and dimensions match.
            if self.fc is not None and self.fc.in_features == self.output_dim:
                 fused = self.fc(fused)

        fused = self.dropout(fused)
        fused = self.activation(fused)
        return fused

    @staticmethod
    def _ensure_2d(tensor):
        if tensor.dim() == 3 and tensor.size(1) == 1:
            return tensor.squeeze(1)
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _ensure_3d(tensor):
        if tensor.dim() == 4 and tensor.size(1) == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"Expected tensor to be 3-D after adjustment, got shape {tensor.shape}")
        return tensor
