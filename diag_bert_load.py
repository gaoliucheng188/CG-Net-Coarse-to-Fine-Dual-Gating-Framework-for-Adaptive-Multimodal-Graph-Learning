"""Fast diagnostic: does local BERT now load (hidden 768) and does the
full_modal checkpoint load into the model without size mismatch?
"""
import sys
import torch
import configs  # noqa: F401  (ensures gpickle shim + config side effects)
from multimodal.text_encoder import TextEncoder

me = TextEncoder(pretrained_model="bert-base-uncased", output_dim=300)
print(f"hidden_size={me.hidden_size} fc_in={me.fc.in_features} fc_out={me.fc.out_features}", flush=True)

ck = torch.load("ckpt/mutag_base_h128_o128_full_modal.pth.tar", map_location="cpu")
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
te = {k: v for k, v in sd.items() if k.startswith("text_encoder.")}
if te:
    k = "text_encoder.fc.weight"
    if k in te:
        print(f"checkpoint text_encoder.fc.weight shape={tuple(te[k].shape)}", flush=True)
    else:
        print("checkpoint has text_encoder keys:", sorted(te)[:6], "...", flush=True)
else:
    print("no text_encoder keys in checkpoint top-level; keys sample:", list(sd)[:10], flush=True)
print("DIAG_DONE", flush=True)