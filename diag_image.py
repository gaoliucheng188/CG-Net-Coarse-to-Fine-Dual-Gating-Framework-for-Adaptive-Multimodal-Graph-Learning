import torch
from data.multimodal_loader import MultimodalDataset
from models_multimodal import ImageEncoder

ml = MultimodalDataset(dataset_name="mutag", gpu=False)
enc = ImageEncoder(output_dim=2048)
print("image_encoder fc out:", enc.fc.out_features)
n_img = len(ml.image_map)
print("n images:", n_img)
feats = []
for nid, path in list(ml.image_map.items())[:8]:
    try:
        f = enc.encode_image(path).squeeze(0)
        feats.append(f)
        print(f"node {nid}: norm={f.norm().item():.3f} min={f.min().item():.3f} max={f.max().item():.3f} "
              f"n_eq0={(f==0).sum().item()}/{f.numel()}")
    except Exception as e:
        print(f"node {nid}: ERR {str(e)[:120]}")
if feats:
    fs = torch.stack(feats)
    # correlation between different nodes' features
    # all distinct?
    d = torch.pdist(fs, p=2)
    print(f"pairwise distances (first 10): {d[:10].tolist()}")
    print("any exact duplicate rows:", (fs.unsqueeze(0)-fs.unsqueeze(1)).abs().max().item()==0)