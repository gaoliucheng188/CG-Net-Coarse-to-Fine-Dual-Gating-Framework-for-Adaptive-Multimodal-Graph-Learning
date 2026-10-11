"""Convert a PyTorch bert .bin checkpoint to .safetensors.

transformers==4.38 + torch==2.0.1 refuse to load .bin state dicts (torch.load
security guard), but will happily load .safetensors. Our full_modal checkpoints
were trained with real BERT (fc input dim 768), so local reproduction needs the
real weights. This makes the local bert-base-uncased folder loadable again.
"""
import torch
from safetensors.torch import save_file

src = "bert-base-uncased/pytorch_model.bin"
dst = "bert-base-uncased/pytorch_model.safetensors"

state = torch.load(src, map_location="cpu")
# state may be a plain OrderedDict of tensors, or have keys/state_dict wrapper
if not isinstance(state, dict) or "state_dict" not in state:
    sd = state
else:
    sd = state["state_dict"]
# Keep only torch.Tensor values (drop any non-tensor metadata). Tied tensors
# (e.g. BERT LM-head decoder weight sharing word-embeddings) share a data_ptr;
# drop the later copy so safetensors has no shared memory.
sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
seen_ptr = {}
unique = {}
for k, v in sd.items():
    ptr = v.untyped_storage().data_ptr() if hasattr(v, "untyped_storage") else v.storage().data_ptr()
    if ptr in seen_ptr:
        print(f"skip tied tensor {k} (shares with {seen_ptr[ptr]})")
        continue
    seen_ptr[ptr] = k
    unique[k] = v.contiguous()
print(f"Converting {len(unique)} tensors -> {dst}")
save_file(unique, dst)
print("Safetensors written OK")