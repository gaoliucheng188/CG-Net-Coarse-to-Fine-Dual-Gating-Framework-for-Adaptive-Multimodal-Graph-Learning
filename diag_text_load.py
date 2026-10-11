from data.multimodal_loader import MultimodalDataset
import numpy as np
import networkx as nx

# Ensure gpickle shim (imported by utils via data module) is active
ml = MultimodalDataset(dataset_name="mutag", gpu=False)
print("text_dir resolved:", ml.text_dir)
print("image_dir resolved:", ml.image_dir)
print("n text entries:", len(ml.text_map))
print("n image entries:", len(ml.image_map))
G = ml.data[0]
print("G nodes:", G.number_of_nodes())
# sample node text
sample = next(iter(G.nodes()))
d = ml.get_node_data(sample)
print("sample node", sample, "text[:50] =", repr(d["text"][:50]), "has_image=", d["has_image"], "img=", d["image_path"])
# count how many distinct texts
vals = set()
for n in G.nodes():
    vals.add(ml.get_node_data(n)["text"])
print("distinct text values across nodes:", len(vals))
for v in list(vals)[:5]:
    print("   ", repr(v[:60]))