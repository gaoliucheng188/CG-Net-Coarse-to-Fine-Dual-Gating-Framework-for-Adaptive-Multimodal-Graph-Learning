import torch
ck = torch.load("ckpt/mutag_base_h128_o128_full_modal.pth.tar", map_location="cpu", weights_only=False)
sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
for k in sd:
    if k.startswith("fusion.") and ("weight" in k or "bias" in k):
        print(f"{k}: {tuple(sd[k].shape)}")
print("pred_model.weight:", tuple(sd["pred_model.weight"].shape) if "pred_model.weight" in sd else "none")
# graph_dim_full = hidden*(layers-1)+output ; layers?
print("conv_first.weight out:", tuple(sd.get("conv_first.weight", [0]).shape))
print("conv_last.weight out:", tuple(sd.get("conv_last.weight", [0]).shape))
for k in sd:
    if k.startswith("conv_block.") and k.endswith(".weight"):
        print(k, tuple(sd[k].shape))