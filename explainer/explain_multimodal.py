
import torch
import torch.nn as nn
import numpy as np
from explainer import explain
from multimodal.text_encoder import TextEncoder
from multimodal.image_encoder import ImageEncoder
import torch.nn.functional as F
import time
import os
from utils import io_utils

def otsu_threshold(mask):
    """
    Compute Otsu's threshold for a given mask (tensor or numpy array).
    Returns the threshold value.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Flatten and normalize to 0-255 for histogram calculation
    mask = mask.flatten()
    if mask.max() == mask.min():
        return mask.max()
        
    # Scale to 8-bit integer range for histogram
    mask_int = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(int)
    hist, bin_edges = np.histogram(mask_int, bins=256, range=(0, 255))
    
    # Normalize histogram
    hist = hist.astype(float) / hist.sum()
    
    # Class probabilities and means
    w0 = np.cumsum(hist)
    w1 = 1.0 - w0
    
    # Handle division by zero
    w0[w0 == 0] = 1e-9
    w1[w1 == 0] = 1e-9
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mu = np.cumsum(hist * bin_centers)
    mu0 = mu / w0
    mu1 = (mu[-1] - mu) / w1
    
    # Between-class variance
    sigma_b_squared = w0 * w1 * (mu0 - mu1) ** 2
    
    # Find max variance
    idx = np.argmax(sigma_b_squared)
    threshold_int = bin_centers[idx]
    
    # Convert back to original scale
    threshold = threshold_int / 255.0 * (mask.max() - mask.min()) + mask.min()
    return threshold

class MultimodalExplainModule(explain.ExplainModule):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        text_feat=None,
        image_feat=None,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
        use_uma=False,
    ):
        super().__init__(adj, x, model, label, args, graph_idx, writer, use_sigmoid, graph_mode)
        self.use_uma = use_uma
        if self.use_uma:
            print("UMA (Unified Modal Attention) Enabled in Explainer.")
        
        self.text_feat = text_feat
        self.image_feat = image_feat
        
        # Initialize Multimodal Masks (M_T, M_I) ~ U(0,1)
        # Scalar masks for modal-level selection (CF-DGF uses this for modal gating)
        # using initialization at 0.0 -> Sigmoid(0) = 0.5 (neutral importance)
        self.mask_text = nn.Parameter(torch.tensor([0.0]))
        self.mask_image = nn.Parameter(torch.tensor([0.0]))
        
        # Re-build optimizer
        params = [self.mask, self.feat_mask, self.mask_text, self.mask_image]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
            
        from utils import train_utils
        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs['size'] = getattr(args, 'lambda_size', 0.005)
        self.coeffs['ent'] = getattr(args, 'lambda_ent', 1.0)

        # Note: Simplified loss function for clarity:
        # L = L_pred + λ_size * L_size + λ_ent * L_ent
        # Removed: L_lap, L_id, L_domain (controversial regularization terms)

        # Hook for capturing intermediate embeddings
        self.captured_feats = None
        self.hook_handle = self.model.fusion.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # input is (graph_feat, text_feat, image_feat)
        self.captured_feats = input

    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        x = self.x.cuda() if self.args.gpu else self.x
        
        # Ensure multimodal features are on the correct device
        device = self.mask_text.device
        if self.text_feat is not None:
            self.text_feat = self.text_feat.to(device)
        if self.image_feat is not None:
            self.image_feat = self.image_feat.to(device)

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                x = x * feat_mask

        m_t = torch.sigmoid(self.mask_text) if self.use_sigmoid else self.mask_text
        m_i = torch.sigmoid(self.mask_image) if self.use_sigmoid else self.mask_image
        
        masked_text_feat = None
        if self.text_feat is not None:
            masked_text_feat = self.text_feat * m_t
            
        masked_image_feat = None
        if self.image_feat is not None:
            masked_image_feat = self.image_feat * m_i

        ypred, adj_att = self.model(
            x, 
            self.masked_adj, 
            text_feats=masked_text_feat, 
            image_feats=masked_image_feat
        )
        
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = self._select_node_logits(ypred, node_idx)
            res = nn.Softmax(dim=0)(node_pred)
            
        return res, adj_att

    def loss(self, pred, pred_label, node_idx, epoch):
        # --- Basic GNNExplainer Loss ---
        # IMPORTANT (test-time fairness): optimize masks to preserve the model's
        # *own predicted class*, never the ground-truth label. Using GT here would
        # leak test labels into Soft Masker fusion weights (s_T).
        # L = L_pred + λ_size * L_size + λ_ent * L_ent
        pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
        if torch.is_tensor(pred_label_node):
            target = int(pred_label_node.detach().cpu().item())
        else:
            target = int(pred_label_node)
        logit = pred[target]
        pred_loss = -torch.log(logit + 1e-10)

        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        mask_ent = -mask * torch.log(mask + 1e-10) - (1 - mask) * torch.log(1 - mask + 1e-10)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        # Modality-sparsity prior: penalize the text weight s_T directly, so text must
        # EARN its influence by raising the graph-only answer's probability
        # (modality-safe fusion). Without it, a hard-class anchor alone does not
        # penalize text that dilutes confidence without flipping the class.
        if self.text_feat is not None:
            lam_mod = getattr(self.args, "lambda_mod", 0.5)
            mod_prior_loss = lam_mod * torch.sigmoid(self.mask_text).sum()
        else:
            mod_prior_loss = torch.tensor(0.0, device=self.mask.device)

        total_loss = pred_loss + size_loss + mask_ent_loss + mod_prior_loss

        if self.writer is not None:
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/total_loss", total_loss, epoch)

        return total_loss

class MultimodalExplainer(explain.Explainer):
    def __init__(self, model, adj, feat, label, pred, train_idx, args, writer=None,
                 print_training=True, graph_mode=False, graph_idx=-1, use_uma=False,
                 pred_graph=None):
        super().__init__(model, adj, feat, label, pred, train_idx, args, writer,
                         print_training, graph_mode, graph_idx, pred_graph=pred_graph)
        self.use_uma = use_uma
        self.disable_text = getattr(args, "disable_text", False)
        self.disable_image = getattr(args, "disable_image", False)

        self.text_encoder = None
        if not self.disable_text:
            self.text_encoder = TextEncoder(
                output_dim=getattr(args, "text_dim", 128),
                gpu=args.gpu,
                force_fallback=getattr(args, "force_text_fallback", False),
            )
        self.image_encoder = None
        if not self.disable_image:
            self.image_encoder = ImageEncoder(
                output_dim=getattr(args, "image_dim", 128),
                gpu=args.gpu,
            )

    def explain(self, node_idx, graph_idx=-1, graph_mode=False, unconstrained=False):
        data_loader = self.args.multimodal_loader
        if graph_mode:
            node_data = data_loader.get_graph_data(graph_idx)
        else:
            node_data = data_loader.get_node_data(node_idx)
            
        text_feat = None
        if not self.disable_text and node_data.get('text'):
            text_feat = self.text_encoder.encode_text([node_data['text']])
            
        image_feat = None
        if not self.disable_image and node_data.get('image_path'):
            image_feat = self.image_encoder.encode_image(node_data['image_path'])

        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        
        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            # Modality-safe fusion: anchor the mask objective on the graph-only
            # (text-disabled) predicted class, so s_T is driven down when text
            # contradicts the reliable topology signal. Full-fusion class is NOT used
            # as target (it is already text-polluted).
            pred_label = self._gather_node_predictions(neighbors, source="graph")

        explainer = MultimodalExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            text_feat=text_feat,
            image_feat=image_feat,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
            use_uma=self.use_uma,
        )
        
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()
        explainer.train()
        begin_time = time.time()
        
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            
            ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            if self.print_training and epoch % 20 == 0:
                print(f"epoch: {epoch}; loss: {loss.item():.4f}")

        # Remove hook
        if hasattr(explainer, 'hook_handle'):
            explainer.hook_handle.remove()

        masked_adj = explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
        masked_adj = self._apply_domain_rules(masked_adj)
        
        threshold = otsu_threshold(masked_adj)
        masked_adj[masked_adj < threshold] = 0.0
        
        # Calculate modality importance scores (mean of the learned masks)
        text_score = torch.sigmoid(explainer.mask_text).mean().item() if explainer.text_feat is not None else 0.0
        image_score = torch.sigmoid(explainer.mask_image).mean().item() if explainer.image_feat is not None else 0.0
        
        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            
        return masked_adj, text_score, image_score

    def _apply_domain_rules(self, mask):
        # Domain Rule: Valency Constraint (Max degree <= 4 for molecules)
        # This is a heuristic post-processing step.
        # We scale down edges for nodes that violate the valency constraint.
        degrees = np.sum(mask, axis=1)
        for i in range(len(degrees)):
            if degrees[i] > 4.0:
                # Scale down edges connected to node i to satisfy sum <= 4
                scale = 4.0 / (degrees[i] + 1e-9)
                mask[i, :] *= scale
                mask[:, i] *= scale
        return mask

