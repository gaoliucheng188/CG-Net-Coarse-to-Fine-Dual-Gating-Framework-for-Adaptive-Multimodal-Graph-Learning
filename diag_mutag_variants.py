import torch, glob, os
for p in sorted(glob.glob("ckpt/mutag_*.pth.tar")):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        # fusion.fc input width, or pred_model input if no fusion.fc
        if "fusion.fc.weight" in sd:
            w = int(sd["fusion.fc.weight"].shape[1])
            tag = f"fusion.fc in={w}"
        else:
            w = int(sd.get("pred_model.weight", torch.zeros(2)).shape[1])
            tag = f"pred_model in={w} (no fusion.fc)"
        pm = int(sd.get("pred_model.weight", torch.zeros(2)).shape[1])
        print(f"{os.path.basename(p):55s} {tag:22s} pred_model in={pm}")
    except Exception as e:
        print(f"{os.path.basename(p):55s} ERR {e}")