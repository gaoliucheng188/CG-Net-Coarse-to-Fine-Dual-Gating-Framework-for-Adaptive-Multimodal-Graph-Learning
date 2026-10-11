import argparse, torch, configs
from models_multimodal import MultimodalGcnEncoderNode

a = configs.arg_parse()
a.multimodal = True
a.hidden_dim = 128; a.output_dim = 128; a.text_dim = 300; a.image_dim = 2048
a.num_gc_layers = 3; a.fusion_method = "concat"; a.bn = False
a.disable_text = False; a.disable_image = False
a.project_concat = True

m = MultimodalGcnEncoderNode(input_dim=71, hidden_dim=128, embedding_dim=128,
    label_dim=2, num_layers=3, bn=False, args=a, text_dim=300, image_dim=2048,
    fusion_method="concat", project_concat=True, disable_text=False, disable_image=False)

print("cuda available:", torch.cuda.is_available())
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = m.to(dev); m.eval()
w = m.text_encoder.bert.embeddings.word_embeddings.weight
print("text_encoder.bert embeddings device:", w.device)
print("text_encoder.fc device:", m.text_encoder.fc.weight.device)
print("conv_first device:", m.conv_first.weight.device)
print("fusion device:", m.fusion.fc.weight.device if hasattr(m.fusion,'fc') else "n/a")