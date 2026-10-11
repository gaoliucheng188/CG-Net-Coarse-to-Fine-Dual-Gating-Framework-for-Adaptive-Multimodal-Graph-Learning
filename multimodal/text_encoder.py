import os
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
)


class TextEncoder(nn.Module):
    """Text encoder with robust fallback for low-memory machines.

    Loading strategy:
      1) Try to load `pretrained_model` (BERT) with low_cpu_mem_usage if supported.
      2) If that fails (OOM / file error), try DistilBERT (smaller).
      3) If DistilBERT also fails, fallback to a hashing-based nn.Embedding encoder.
    """

    def __init__(self, pretrained_model="bert-base-uncased", output_dim=128, gpu=True, max_length=128, force_fallback=False):
        super().__init__()
        self.gpu = gpu
        self.max_length = max_length
        self.is_distil = False
        self.is_fallback = False
        self.vocab_size = 20000  # for hashing fallback
        self.hidden_size = None
        # flag indicating we created a tokenizer-based embedding fallback
        self.tokenizer_embedding = False

        # If caller forces fallback, enable lightweight embedding immediately
        if force_fallback:
            # prefer tokenizer-based embedding if tokenizer files exist, otherwise hashing
            if os.path.isdir(pretrained_model):
                # try load tokenizer local-only
                try:
                    self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, local_files_only=True)
                except Exception:
                    self.tokenizer = None
            else:
                self.tokenizer = None

            if getattr(self, 'tokenizer', None) is not None:
                try:
                    vocab_len = getattr(self.tokenizer, 'vocab_size', None) or len(getattr(self.tokenizer, 'vocab', {}))
                    if not vocab_len:
                        vocab_len = self.vocab_size
                    self.embedding = nn.Embedding(vocab_len, 300)
                    self.hidden_size = 300
                    self.is_fallback = True
                    self.tokenizer_embedding = True
                    print("Info: force_fallback enabled: using tokenizer-vocab embedding")
                except Exception:
                    self.is_fallback = True
                    self.hidden_size = 300
                    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
            else:
                self.is_fallback = True
                self.hidden_size = 300
                self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # Helper: detect if local weight files (safetensors or pytorch bin) exist
        def _has_local_weights(path):
            try:
                if os.path.isdir(path):
                    safetensors_path = os.path.join(path, "pytorch_model.safetensors")
                    bin_path = os.path.join(path, "pytorch_model.bin")
                    return os.path.exists(safetensors_path) or os.path.exists(bin_path)
            except Exception:
                return False
            return False

        def _has_safetensors(path):
            try:
                if os.path.isdir(path):
                    safetensors_path = os.path.join(path, "pytorch_model.safetensors")
                    return os.path.exists(safetensors_path)
            except Exception:
                return False
            return False

        def _has_any_weights(path):
            try:
                if os.path.isdir(path):
                    safetensors_path = os.path.join(path, "pytorch_model.safetensors")
                    bin_path = os.path.join(path, "pytorch_model.bin")
                    return os.path.exists(safetensors_path) or os.path.exists(bin_path)
            except Exception:
                return False
            return False

        # Helper: detect if local tokenizer files exist in path
        def _has_local_tokenizer(path):
            try:
                if os.path.isdir(path):
                    for fname in ["vocab.txt", "tokenizer.json", "vocab.json", "tokenizer_config.json", "merges.txt"]:
                        if os.path.exists(os.path.join(path, fname)):
                            return True
            except Exception:
                return False
            return False

        # Quick PyTorch version parse (major, minor)
        try:
            ver = torch.__version__.split("+")[0]
            major, minor = map(int, ver.split('.')[:2])
        except Exception:
            major, minor = 0, 0

        # Try to load tokenizer first (local-only if available). We do this early so
        # that if model weights can't be safely loaded (torch<2.6 and no safetensors)
        # we can still fall back to a tokenizer-vocab embedding without attempting
        # to download or load transformer weights.
        local_tokenizer_present = False
        try:
            local_tokenizer_present = _has_local_tokenizer(pretrained_model)
            if local_tokenizer_present:
                self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, local_files_only=True)
            else:
                # load tokenizer from remote/cache if available
                self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        except Exception:
            try:
                if _has_local_tokenizer("distilbert-base-uncased"):
                    self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
                else:
                    self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            except Exception:
                self.tokenizer = None

        # If PyTorch is older than 2.6 and there are no safetensors weights locally,
        # do NOT attempt to load transformer model weights (this triggers the
        # torch.load security restriction). Instead, prefer tokenizer-vocab embedding
        # (if tokenizer available) or the hashing fallback.
        if (major, minor) < (2, 6) and not _has_safetensors(pretrained_model):
            if _has_any_weights(pretrained_model):
                print(
                    "Info: Found local PyTorch model weights (pytorch_model.bin) but PyTorch < 2.6."
                    " Loading those with current PyTorch is blocked for security."
                )
            else:
                print(
                    "Info: Detected PyTorch < 2.6 and no local transformer weights. "
                    "Skipping transformer model loads and using hashing fallback."
                )
            # Use tokenizer-based embedding if tokenizer exists, otherwise hashing fallback
            if getattr(self, 'tokenizer', None) is not None:
                try:
                    vocab_len = getattr(self.tokenizer, 'vocab_size', None) or len(getattr(self.tokenizer, 'vocab', {}))
                    if not vocab_len:
                        vocab_len = self.vocab_size
                    self.embedding = nn.Embedding(vocab_len, 300)
                    self.hidden_size = 300
                    self.is_fallback = True
                    self.tokenizer_embedding = True
                    print("Info: Using tokenizer-vocab embedding fallback (no model weights loaded).")
                except Exception:
                    self.is_fallback = True
                    self.hidden_size = 300
                    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
            else:
                self.is_fallback = True
                self.hidden_size = 300
                self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # try to load tokenizer first (usually small) -- only if not in early fallback
        local_tokenizer_present = False
        if not self.is_fallback:
            try:
                local_tokenizer_present = _has_local_tokenizer(pretrained_model)
                if local_tokenizer_present:
                    # prefer local-only tokenizer load to avoid network
                    self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, local_files_only=True)
                else:
                    self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            except Exception:
                # fallback to DistilBert tokenizer if Bert tokenizer loading fails
                try:
                    # try local-only first
                    if _has_local_tokenizer("distilbert-base-uncased"):
                        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
                    else:
                        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                except Exception:
                    # tokenizer might not be available; we'll use a simple whitespace tokenizer in fallback
                    self.tokenizer = None

        # If tokenizer exists locally and there are no local model weights, use tokenizer-based embedding fallback
        if getattr(self, 'tokenizer', None) is not None and not _has_local_weights(pretrained_model):
            try:
                vocab_len = getattr(self.tokenizer, 'vocab_size', None) or len(getattr(self.tokenizer, 'vocab', {}))
                if not vocab_len:
                    vocab_len = self.vocab_size
                self.embedding = nn.Embedding(vocab_len, 300)
                self.hidden_size = 300
                self.is_fallback = True
                self.tokenizer_embedding = True
                print("Info: Found local tokenizer but no model weights; using tokenizer-vocab embedding fallback (no network calls).")
            except Exception:
                # If this fails, continue to model load attempts which may raise informative errors
                pass

        # try loading BERT model with low memory option
        if not self.is_fallback:
            try:
                # Prefer loading from local files when available to avoid network calls
                local_only = False
                # If user passed a local path, force local_files_only
                if os.path.isdir(pretrained_model):
                    local_only = True

                # If a safetensors file exists in a local folder, prefer local loading
                safetensors_path = os.path.join(pretrained_model, "pytorch_model.safetensors")
                bin_path = os.path.join(pretrained_model, "pytorch_model.bin")
                if os.path.exists(safetensors_path) or os.path.exists(bin_path):
                    local_only = True

                # low_cpu_mem_usage requires the `accelerate` package and raises a
                # non-TypeError when it is missing, so try a plain local load first
                # (works with the local .safetensors, no network, no accelerate).
                try:
                    self.bert = BertModel.from_pretrained(pretrained_model, local_files_only=local_only)
                except TypeError:
                    # very old transformers may need low_cpu_mem_usage off explicitly
                    self.bert = BertModel.from_pretrained(pretrained_model, local_files_only=local_only)

                self.hidden_size = getattr(self.bert.config, "hidden_size", 768)
                self.is_distil = False
            except Exception as e1:
                # Detect common causes and provide actionable messages
                msg = str(e1)
                print("Warning: BERT load failed (will try DistilBERT). Error:\n", msg)
                if "torch.load" in msg or "vulnerability" in msg or "weights_only" in msg:
                    print(
                        "Detected torch.load security restriction. Transformers requires torch>=2.6 or models in safetensors format to avoid this.\n"
                        "Options:\n  1) Upgrade PyTorch to >=2.6 that matches your CUDA (recommended).\n  2) Download model files in safetensors format (pytorch_model.safetensors) or the model folder into './bert-base-uncased/' and retry.\n"
                    )

                # try DistilBERT
                try:
                    # Try loading DistilBERT from local cache first
                    distil_local = False
                    if os.path.isdir("distilbert-base-uncased"):
                        distil_local = True
                    try:
                        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=distil_local) if self.tokenizer is None else self.tokenizer
                        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", local_files_only=distil_local)
                    except TypeError:
                        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") if self.tokenizer is None else self.tokenizer
                        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                    self.hidden_size = getattr(self.bert.config, "hidden_size", 768)
                    self.is_distil = True
                    print("Info: Loaded DistilBERT as fallback.")
                except Exception as e2:
                    print("Warning: DistilBERT load also failed. Falling back to hashing embedding encoder. Error:\n", e2)
                    # fallback: hashing embedding encoder (lightweight)
                    self.is_fallback = True
                    self.hidden_size = 300
                    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # If we reached here without loading a transformer model but we do have a tokenizer
        # (e.g., local tokenizer files exist but model weights are missing), create a
        # tokenizer-based embedding as an automatic fix: this uses the tokenizer's vocab
        # size to create a trainable embedding and maps tokens -> ids via tokenizer.
        if not hasattr(self, 'bert') and getattr(self, 'tokenizer', None) is not None and not self.is_fallback:
            try:
                vocab_len = getattr(self.tokenizer, 'vocab_size', None) or len(getattr(self.tokenizer, 'vocab', {}))
                # some tokenizers expose 'vocab' mapping; fallback to 300 if unknown
                if vocab_len is None or vocab_len == 0:
                    vocab_len = self.vocab_size
                self.embedding = nn.Embedding(vocab_len, 300)
                self.hidden_size = 300
                self.is_fallback = True
                self.tokenizer_embedding = True
                print("Info: No transformer weights found but tokenizer exists; using tokenizer-vocab embedding fallback.")
            except Exception:
                # leave as-is; earlier fallback will handle
                pass

        # fully connected projection defined after determining hidden_size
        if self.hidden_size is None:
            # last resort
            self.hidden_size = 300
            self.is_fallback = True
            self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.fc = nn.Linear(self.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()

        # Do not move module to CUDA here. Movement should be handled by caller via
        # model.to(device). We keep self.gpu as a preference flag.

    def forward(self, texts):
        """Encode a batch (list) of text strings and return [B, output_dim] tensor."""
        # hashing / tokenizer-based fallback path
        if self.is_fallback:
            # If we created a tokenizer-based embedding, use tokenizer -> ids -> embedding
            if getattr(self, 'tokenizer_embedding', False) and getattr(self, 'tokenizer', None) is not None:
                # tokenize to ids (avoid attention masks etc.; just get input ids)
                enc = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                ids_tensor = enc['input_ids']
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device('cpu')
                ids_tensor = ids_tensor.to(device)
                emb = self.embedding(ids_tensor)  # [B, L, H]
                # simple average pooling
                feat = emb.mean(dim=1)
                x = self.dropout(feat)
                x = self.fc(x)
                x = self.activation(x)
                return x

            # otherwise fallback to simple whitespace hashing embedding
            batch_ids = []
            max_len = min(self.max_length, max((len(t.split()) for t in texts), default=1))
            for t in texts:
                toks = t.split()[:max_len]
                ids = [abs(hash(w)) % self.vocab_size for w in toks]
                # pad
                if len(ids) < max_len:
                    ids = ids + [0] * (max_len - len(ids))
                batch_ids.append(ids)
            ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
            # move ids to same device as module parameters
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            ids_tensor = ids_tensor.to(device)
            emb = self.embedding(ids_tensor)  # [B, L, H]
            # average pooling (masking ignored for simplicity)
            feat = emb.mean(dim=1)
            x = self.dropout(feat)
            x = self.fc(x)
            x = self.activation(x)
            return x

        # normal (BERT / DistilBERT) path
        # tokenizer should be present
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer available for text encoding")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # move inputs to same device as module
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.bert(**inputs)
        # use [CLS] token embedding if available; otherwise use first token
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            # some models return tuple
            last_hidden = outputs[0]
        cls_feat = last_hidden[:, 0, :]
        x = self.dropout(cls_feat)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def encode_text(self, texts):
        self.eval()
        with torch.no_grad():
            return self.forward(texts)