import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GraphConv, GcnEncoderGraph
from multimodal.text_encoder import TextEncoder
from multimodal.image_encoder import ImageEncoder
from multimodal.fusion import MultimodalFusion

class MultimodalGcnEncoderNode(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 text_dim=128, image_dim=128, fusion_method='concat', **kwargs):
        force_fallback = kwargs.pop('force_text_fallback', False)
        disable_text = kwargs.pop('disable_text', False)
        disable_image = kwargs.pop('disable_image', False)
        project_concat = kwargs.pop('project_concat', True)
        super().__init__(input_dim, hidden_dim, embedding_dim, label_dim, num_layers, **kwargs)

        # gcn_forward outputs features with dimensionality self.pred_input_dim
        graph_feature_dim = self.pred_input_dim

        # graph_feat 的最后一维是 hidden_dim
        self.text_encoder = TextEncoder(output_dim=text_dim, force_fallback=force_fallback)
        self.image_encoder = ImageEncoder(output_dim=image_dim)
        self.fusion = MultimodalFusion(
            graph_dim=graph_feature_dim,
            text_dim=text_dim,
            image_dim=image_dim,
            output_dim=embedding_dim,
            fusion_method=fusion_method,
            disable_text=disable_text,
            disable_image=disable_image,
            project_concat=project_concat,
        )
        self._text_cache = {}
        self._image_cache = {}

        # 调整预测层输入维度
        if fusion_method == 'concat' and not project_concat:
            self.pred_input_dim = graph_feature_dim
            if not disable_text:
                self.pred_input_dim += text_dim
            if not disable_image:
                self.pred_input_dim += image_dim
        else:
            self.pred_input_dim = embedding_dim
            
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, kwargs.get('pred_hidden_dims', []), label_dim
        )

    def forward(self, x, adj, text=None, image_path=None, batch_num_nodes=None, **kwargs):
        # Support for calling with text_feats/image_feats (compatibility with train.py)
        if text is None:
            text = kwargs.get('text_feats')
        if image_path is None:
            image_path = kwargs.get('image_feats')

        # 原始GCN特征
        graph_feat, adj_att = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last)
        # graph_feat: [batch, num_nodes, graph_feature_dim]

        # 文本特征
        text_feat = torch.zeros(graph_feat.size(0), graph_feat.size(1), self.text_encoder.fc.out_features, device=graph_feat.device)
        if text is not None:
            # Check if text is a tensor (pre-encoded) or list (raw text)
            if isinstance(text, torch.Tensor):
                # If it's a tensor, assume it's already encoded or needs simple projection
                # The explainer passes masked embeddings.
                if text.dim() == 3 and text.size(2) == self.text_encoder.fc.out_features:
                     text_feat = text
                elif text.dim() == 2 and text.size(1) == self.text_encoder.fc.out_features:
                     # Broadcast [batch, dim] to [batch, num_nodes, dim]
                     text_feat = text.unsqueeze(1).expand(-1, graph_feat.size(1), -1)
                else:
                     # Fallback or error?
                     pass
            else:
                text_emb = self._lookup_text_embeddings(text, graph_feat.size(1), graph_feat.device)
                if text_emb is not None:
                    text_feat[:, :text_emb.size(0), :] = text_emb.unsqueeze(0)

        # 图像特征
        image_feat = torch.zeros(graph_feat.size(0), graph_feat.size(1), self.image_encoder.fc.out_features, device=graph_feat.device)
        if image_path is not None:
             if isinstance(image_path, torch.Tensor):
                 if image_path.dim() == 3 and image_path.size(2) == self.image_encoder.fc.out_features:
                     image_feat = image_path
                 elif image_path.dim() == 2 and image_path.size(1) == self.image_encoder.fc.out_features:
                     image_feat = image_path.unsqueeze(1).expand(-1, graph_feat.size(1), -1)
             else:
                image_emb = self._lookup_image_embeddings(image_path, graph_feat.size(1), graph_feat.device)
                if image_emb is not None:
                    image_feat[:, :image_emb.size(0), :] = image_emb.unsqueeze(0)

        # 融合特征
        fused_feat = self.fusion(graph_feat, text_feat, image_feat)
        # fused_feat: [batch, num_nodes, fusion_dim]

        # 预测：对每个节点做分类
        # 如果batch=1，去掉batch维
        if fused_feat.size(0) == 1:
            node_feat = fused_feat[0]  # [num_nodes, fusion_dim]
        else:
            node_feat = fused_feat.view(-1, fused_feat.size(-1))
        pred = self.pred_model(node_feat)  # [num_nodes, num_classes]
        return pred, adj_att

    def _lookup_text_embeddings(self, text_list, max_nodes, device):
        if not text_list:
            return None
        keys = []
        for i in range(min(max_nodes, len(text_list))):
            val = text_list[i]
            keys.append(val if isinstance(val, str) else "")
        if not keys:
            return None
        missing = []
        seen = set()
        for key in keys:
            if key not in self._text_cache and key not in seen:
                missing.append(key)
                seen.add(key)
        if missing:
            enc = self.text_encoder.encode_text(missing)
            for key, emb in zip(missing, enc):
                self._text_cache[key] = emb.detach().cpu()
        stacked = torch.stack([self._text_cache.get(key, torch.zeros(self.text_encoder.fc.out_features)) for key in keys])
        return stacked.to(device)

    def _lookup_image_embeddings(self, paths, max_nodes, device):
        if not paths:
            return None
        keys = []
        for i in range(min(max_nodes, len(paths))):
            path = paths[i]
            keys.append(path if isinstance(path, str) and path else None)
        if not keys:
            return None
        for key in keys:
            if key and key not in self._image_cache:
                emb = self.image_encoder.encode_image(key).squeeze(0).detach().cpu()
                self._image_cache[key] = emb
        stacked = []
        zero = torch.zeros(self.image_encoder.fc.out_features)
        for key in keys:
            if key is None:
                stacked.append(zero)
            else:
                stacked.append(self._image_cache[key])
        return torch.stack(stacked).to(device)

class MultimodalGcnEncoderGraph(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 text_dim=128, image_dim=128, fusion_method='concat', **kwargs):
        force_fallback = kwargs.pop('force_text_fallback', False)
        disable_text = kwargs.pop('disable_text', False)
        disable_image = kwargs.pop('disable_image', False)
        project_concat = kwargs.pop('project_concat', True)
        pred_hidden_dims = kwargs.get('pred_hidden_dims', [])
        super().__init__(input_dim, hidden_dim, embedding_dim, label_dim, num_layers, **kwargs)

        graph_feature_dim = self.pred_input_dim
        # print(f"DEBUG: graph_feature_dim={graph_feature_dim}, text_dim={text_dim}, image_dim={image_dim}, fusion={fusion_method}, project={project_concat}")

        self.text_encoder = TextEncoder(output_dim=text_dim, force_fallback=force_fallback)
        self.image_encoder = ImageEncoder(output_dim=image_dim)
        self.fusion = MultimodalFusion(
            graph_dim=graph_feature_dim,
            text_dim=text_dim,
            image_dim=image_dim,
            output_dim=embedding_dim,
            fusion_method=fusion_method,
            disable_text=disable_text,
            disable_image=disable_image,
            project_concat=project_concat,
        )

        # 调整预测层输入维度
        if fusion_method == 'concat' and not project_concat:
            self.pred_input_dim = graph_feature_dim
            if not disable_text:
                self.pred_input_dim += text_dim
            if not disable_image:
                self.pred_input_dim += image_dim
        else:
            self.pred_input_dim = embedding_dim
            
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim
        )

    def forward(self, x, adj, text="", image_path=None, batch_num_nodes=None, **kwargs):
        # Support for calling with text_feats/image_feats (compatibility with train.py)
        if text == "" and 'text_feats' in kwargs:
            text = kwargs['text_feats']
        if image_path is None and 'image_feats' in kwargs:
            image_path = kwargs['image_feats']

        # 原始GCN特征
        graph_feat, adj_att = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last)
        
        # Fix: gcn_forward always returns concatenated features. 
        # If concat is False, we should only use the last layer's output.
        if hasattr(self, 'concat') and not self.concat:
            last_dim = self.conv_last.output_dim
            graph_feat = graph_feat[:, :, -last_dim:]
            
        graph_feat_global = torch.max(graph_feat, dim=1)[0]  # 图级特征

        # 文本特征
        text_feat = self.text_encoder.encode_text([text])

        # 图像特征
        image_feat = torch.zeros(1, self.image_encoder.fc.out_features, device=graph_feat.device)
        if image_path:
            image_feat = self.image_encoder.encode_image(image_path)

        # 融合特征
        fused_feat = self.fusion(graph_feat_global, text_feat, image_feat)

        # 预测
        pred = self.pred_model(fused_feat)
        return pred


