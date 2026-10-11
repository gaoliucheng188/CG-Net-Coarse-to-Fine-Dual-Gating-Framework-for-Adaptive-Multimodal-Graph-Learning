import argparse
import os
import pickle
import re
import shutil
import textwrap
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np


def hash_embed(text: str, dim: int = 300) -> np.ndarray:
    """Simple hashing-based embedding (bag-of-words) with deterministic seed."""
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    for tok in tokens:
        slot = hash(tok) % dim
        vec[slot] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def extract_doc_text(root: ET.Element) -> str:
    parts = []
    for tag in [
        ".//article-title",
        ".//abstract",
        ".//title",
        ".//paragraph",
        ".//body",
        ".//passage/text",
    ]:
        for node in root.findall(tag):
            segment = " ".join((node.text or "").split())
            if segment:
                parts.append(segment)
    text = " ".join(parts)
    return text.strip()


def extract_chemical_mentions(root: ET.Element) -> set:
    chems = set()
    for ann in root.findall(".//annotation"):
        infons = {inf.attrib.get("key", "").lower(): (inf.text or "") for inf in ann.findall("infon")}
        mention_text = (ann.findtext("text") or "").strip()
        ann_type = infons.get("type", "").lower()
        identifier = infons.get("identifier") or infons.get("id") or ""
        if "chem" in ann_type:
            if identifier:
                chems.add(identifier.strip())
            elif mention_text:
                chems.add(mention_text.lower())
    # fallback: look for elements whose tag contains "chemical"
    if not chems:
        for node in root.iter():
            if "chem" in node.tag.lower():
                txt = " ".join((node.text or "").split())
                if txt:
                    chems.add(txt.lower())
    return chems


def bucket_label(count: int) -> int:
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    if count <= 5:
        return 2
    if count <= 10:
        return 3
    return 4


def load_split_ids(raw_root: Path) -> dict:
    split_map = {}
    for split in ["train", "dev", "test"]:
        file_path = raw_root / f"pmcids_{split}.txt"
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                split_map[line] = split
    return split_map


def prepare_dataset(raw_root: Path, output_root: Path, text_dim: int = 300):
    xml_dir = raw_root / "ALL"
    if not xml_dir.exists():
        raise FileNotFoundError(f"Expected XML directory at {xml_dir}")

    split_map = load_split_ids(raw_root)

    docs = []
    for xml_file in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as exc:
            print(f"[WARN] Skipping {xml_file.name}: parse error {exc}")
            continue
        root = tree.getroot()
        pmcid_match = re.match(r"(\d+)", xml_file.stem)
        pmcid = pmcid_match.group(1) if pmcid_match else xml_file.stem
        text = extract_doc_text(root)
        chems = extract_chemical_mentions(root)
        count = len(chems)
        label = bucket_label(count)
        split = split_map.get(pmcid, "train")
        feature = hash_embed(text, dim=text_dim)
        docs.append(
            {
                "pmcid": pmcid,
                "text": text,
                "chemicals": sorted(chems),
                "chem_count": count,
                "label": label,
                "split": split,
                "feature": feature,
            }
        )

    if not docs:
        raise RuntimeError("No documents were parsed from the XML files.")

    # assign sequential node ids
    for idx, doc in enumerate(docs):
        doc["node_id"] = idx

    G = nx.Graph()
    for doc in docs:
        G.add_node(
            doc["node_id"],
            pmcid=doc["pmcid"],
            feat=doc["feature"],
            text=doc["text"],
            chem_count=doc["chem_count"],
            split=doc["split"],
        )

    # Build lightweight edges by connecting each node to k nearest neighbours based on chem_count
    sorted_docs = sorted(docs, key=lambda d: d["chem_count"])
    k = 3
    for idx, doc in enumerate(sorted_docs):
        for offset in range(1, k + 1):
            if idx + offset < len(sorted_docs):
                neighbour = sorted_docs[idx + offset]
                G.add_edge(doc["node_id"], neighbour["node_id"])

    labels = [doc["label"] for doc in sorted(docs, key=lambda d: d["node_id"])]
    num_classes = len(set(labels))

    output_root.mkdir(parents=True, exist_ok=True)
    processed_dir = output_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed_dir / "nlm_chem_graph.gpickle"
    labels_path = processed_dir / "nlm_chem_labels.csv"
    num_classes_path = processed_dir / "num_classes.txt"

    nx.write_gpickle(G, graph_path)
    np.savetxt(labels_path, np.array(labels, dtype=int), fmt="%d")
    with open(num_classes_path, "w", encoding="utf-8") as fh:
        fh.write(str(num_classes))

    # Save pickle for MultimodalDataset compatibility
    pkl_path = output_root / "nlm_chem.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump((G, labels, "nlm_chem"), fh)
    top_level_pkl = Path("data") / "nlm_chem.pkl"
    if top_level_pkl.resolve() != pkl_path.resolve():
        top_level_pkl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pkl_path, top_level_pkl)

    # Dump text files for MultimodalDataset
    text_dir = Path("data/texts/nlm_chem")
    text_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        node_file = text_dir / f"node_{doc['node_id']}.txt"
        content = doc["text"] or "No description available."
        with open(node_file, "w", encoding="utf-8") as fh:
            fh.write(content)

    summary = textwrap.dedent(
        f"""
        Prepared NLM-Chem dataset with {len(docs)} documents.
        Graph saved to: {graph_path}
        Labels saved to: {labels_path}
        Classes: {num_classes}
        Text features directory: {text_dir}
        Top-level pickle: {top_level_pkl}
        """
    ).strip()
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Prepare NLM-Chem corpus for multimodal GNN training")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/nlm_chem/raw/FINAL_v1",
        help="Directory containing the original NLM-Chem corpus (FINAL_v1)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/nlm_chem",
        help="Directory where processed artifacts will be stored",
    )
    parser.add_argument(
        "--text-dim",
        type=int,
        default=300,
        help="Dimension of hashed text features",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw NLM-Chem directory not found: {raw_root}")

    prepare_dataset(raw_root, output_root, text_dim=args.text_dim)


if __name__ == "__main__":
    main()
