"""
Precompute text embeddings for nodes and save as numpy array per dataset.

Usage example:
python scripts/precompute_text_embeddings.py --dataset pqaref --data-dir data --model distilbert-base-uncased --batch-size 32 --outdir data/texts_embeddings

Outputs:
- data/texts_embeddings/<dataset>/embeddings.npy  (shape [N, D])
- data/texts_embeddings/<dataset>/mapping.txt     (node_id per line)

If any node text is missing, a zero vector is used.
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizer, DistilBertModel


def load_texts(text_dir):
    texts = {}
    if not os.path.exists(text_dir):
        return texts
    for fname in os.listdir(text_dir):
        if not fname.startswith('node_') or not fname.endswith('.txt'):
            continue
        try:
            nid = int(fname.split('.')[0].split('_')[1])
        except Exception:
            continue
        with open(os.path.join(text_dir, fname), 'r', encoding='utf-8') as f:
            texts[nid] = f.read().strip()
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--model', default='distilbert-base-uncased')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--outdir', default='data/texts_embeddings')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    text_dir = os.path.join(args.data_dir, 'texts', args.dataset)
    texts = load_texts(text_dir)
    if len(texts) == 0:
        print('No texts found at', text_dir)
        return

    max_id = max(texts.keys())
    N = max_id + 1

    # load model
    tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    model = DistilBertModel.from_pretrained(args.model)
    model.eval()
    if args.gpu:
        model = model.cuda()

    embedding_dim = model.config.hidden_size
    embeddings = np.zeros((N, embedding_dim), dtype=np.float32)

    node_ids = sorted(list(texts.keys()))
    batch = []
    batch_ids = []
    for nid in tqdm(node_ids, desc='Nodes'):
        txt = texts.get(nid, "")
        batch.append(txt)
        batch_ids.append(nid)
        if len(batch) >= args.batch_size:
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            if args.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outs = model(**inputs)
                last = outs.last_hidden_state.cpu().numpy()  # [B, L, H]
            # take CLS-like vector (first token)
            feats = last[:, 0, :]
            for i, _nid in enumerate(batch_ids):
                embeddings[_nid] = feats[i]
            batch = []
            batch_ids = []
    # leftover
    if len(batch) > 0:
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        if args.gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outs = model(**inputs)
            last = outs.last_hidden_state.cpu().numpy()
        feats = last[:, 0, :]
        for i, _nid in enumerate(batch_ids):
            embeddings[_nid] = feats[i]

    outdir = os.path.join(args.outdir, args.dataset)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'embeddings.npy'), embeddings)
    with open(os.path.join(outdir, 'mapping.txt'), 'w', encoding='utf-8') as f:
        for i in range(N):
            f.write(str(i) + '\n')
    print('Saved embeddings to', outdir)


if __name__ == '__main__':
    main()
