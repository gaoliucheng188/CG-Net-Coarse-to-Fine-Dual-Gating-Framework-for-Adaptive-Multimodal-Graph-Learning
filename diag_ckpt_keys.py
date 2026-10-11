import torch
ck = torch.load("ckpt/mutag_base_h128_o128_full_modal.pth.tar", map_location="cpu", weights_only=False)
sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
text_keys = [k for k in sd if k.startswith("text_encoder.")]
bert_keys = [k for k in text_keys if "bert" in k.lower()]
other_text = [k for k in text_keys if "bert" not in k.lower()]
print(f"total keys={len(sd)}")
print(f"text_encoder.bert.* count={len(bert_keys)}")
if bert_keys:
    print("sample bert keys:", bert_keys[:5])
print(f"text_encoder (non-bert) count={len(other_text)}")
print("sample non-bert:", other_text[:8])
print("has text_encoder.fc.weight:", "text_encoder.fc.weight" in sd)