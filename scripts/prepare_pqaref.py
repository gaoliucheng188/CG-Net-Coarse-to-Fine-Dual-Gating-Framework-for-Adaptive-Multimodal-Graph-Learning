#!/usr/bin/env python3
"""Utility to convert the PQAref parquet splits into a graph dataset that
integrates with the existing training pipeline.

The converter expects the Hugging Face parquet exports (train/validation/test)
to be placed under ``data/pqaref/raw`` by default. It will

* read the three splits
* infer yes/no/maybe labels from the answer field
* build TF-IDF features for every QA item
* construct a k-NN similarity graph between questions
* save the artefacts under ``data/pqaref/processed`` and ``data/texts/pqaref``
* generate ``data/pqaref.pkl`` for the multimodal loader
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATASET_NAME = "pqaref"
DEFAULT_RAW_DIR = Path("data") / DATASET_NAME / "raw"
DEFAULT_PROCESSED_DIR = Path("data") / DATASET_NAME / "processed"
DEFAULT_TEXT_DIR = Path("data") / "texts" / DATASET_NAME
DEFAULT_PKL_PATH = Path("data") / f"{DATASET_NAME}.pkl"

ANSWER_MAP = {
    "yes": 0,
    "no": 1,
    "maybe": 2,
}

ABSTRACT_KEYS = (
    "abstract",
    "passage",
    "chunk",
    "context",
    "text",
    "title",
    "content",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PQAref graph dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing the parquet splits (default: data/pqaref/raw)",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Optional explicit path to the train parquet file.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="Optional explicit path to the validation parquet file.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Optional explicit path to the test parquet file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory where processed files are written (default: data/)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=1024,
        help="Maximum TF-IDF feature dimension (default: 1024)",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help="k in kNN graph construction (default: 8)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Optional cosine similarity threshold for edges (0 disables filtering)",
    )
    return parser.parse_args()


def ensure_pyarrow_available() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - guidance for user
        raise RuntimeError(
            "pyarrow is required to read parquet files. Please install it via 'pip install pyarrow'."
        ) from exc


def discover_split(path: Optional[Path], raw_dir: Path, pattern: str) -> Path:
    if path is not None:
        return path
    matches = sorted(raw_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find parquet file matching pattern '{pattern}' in {raw_dir}")
    if len(matches) > 1:
        print(f"Warning: multiple files matched '{pattern}'. Using {matches[0].name}.")
    return matches[0]


def flatten_abstracts(entry) -> str:
    if entry is None or (isinstance(entry, float) and np.isnan(entry)):
        return ""
    if isinstance(entry, str):
        return entry
    parts: List[str] = []
    if isinstance(entry, (list, tuple)):
        for item in entry:
            parts.append(flatten_abstracts(item))
        return " ".join(filter(None, parts))
    if isinstance(entry, dict):
        collected: List[str] = []
        for key in ABSTRACT_KEYS:
            value = entry.get(key)
            if isinstance(value, str):
                collected.append(value)
        if not collected:
            # fallback: stringify remaining items that look textual
            for value in entry.values():
                if isinstance(value, str):
                    collected.append(value)
        return " ".join(filter(None, collected))
    # fall back to JSON representation for unknown structures
    try:
        return json.dumps(entry, ensure_ascii=False)
    except Exception:
        return str(entry)


def infer_label(answer: Optional[str]) -> Optional[int]:
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    if not text:
        return None
    first_token = text.split()[0].lower().strip(".,:;!?()[]{}\"'")
    if first_token in ANSWER_MAP:
        return ANSWER_MAP[first_token]
    text_lower = text.lower()
    for key, idx in ANSWER_MAP.items():
        if text_lower.startswith(key):
            return idx
    return None


def parse_compound_column(df: pd.DataFrame) -> pd.DataFrame:
    """处理仅包含 `input` 列的情况，将其拆分为 instruction/abstracts/answer."""

    if "input" not in df.columns:
        return df

    def _parse_entry(text: str) -> Dict[str, str]:
        instruction = ""
        answer = ""
        abstract_chunks: List[str] = []
        current_state = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Instruction:"):
                instruction = line[len("Instruction:"):].strip()
                current_state = "instruction"
                continue
            if line.startswith("Abstracts:"):
                current_state = "abstract"
                continue
            if line.startswith("Answer:"):
                answer = line[len("Answer:"):].strip()
                current_state = "answer"
                continue
            if line.startswith("abstract_id:"):
                abstract_chunks.append(line)
                current_state = "abstract"
                continue

            if current_state == "abstract" and abstract_chunks:
                abstract_chunks[-1] = abstract_chunks[-1] + "\n" + raw_line.strip("\n")
            elif current_state == "answer":
                answer = (answer + " " + line).strip()

        return {
            "instruction": instruction,
            "abstracts": "\n\n".join(abstract_chunks),
            "answer": answer,
        }

    parsed = df["input"].astype(str).apply(_parse_entry)
    expanded = pd.DataFrame(parsed.tolist())
    return pd.concat([df.drop(columns=["input"]), expanded], axis=1)


def build_feature_matrix(texts: Iterable[str], max_features: int) -> Tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix.astype(np.float32).toarray(), vectorizer


def build_graph(features: np.ndarray, neighbors: int, min_similarity: float) -> nx.Graph:
    if features.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional")
    num_nodes = features.shape[0]
    graph = nx.Graph()
    for idx in range(num_nodes):
        graph.add_node(idx, feat=features[idx])
    if num_nodes <= 1:
        return graph

    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, num_nodes), metric="cosine", algorithm="brute")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    for src, (dist_row, nbr_row) in enumerate(zip(distances, indices)):
        for dist, dst in zip(dist_row[1:], nbr_row[1:]):  # skip self neighbour at position 0
            sim = 1.0 - dist
            if dst == src:
                continue
            if sim < min_similarity:
                continue
            graph.add_edge(src, int(dst), weight=float(sim))
    return graph


def write_text_blobs(texts: List[str], answers: List[str], abstracts: List[str], text_dir: Path) -> None:
    text_dir.mkdir(parents=True, exist_ok=True)
    template = "Instruction: {instruction}\n\nAnswer: {answer}\n\nAbstract Context:\n{abstracts}\n"
    for idx, (instruction, answer, abstract) in enumerate(zip(texts, answers, abstracts)):
        content = template.format(
            instruction=instruction or "",
            answer=answer or "",
            abstracts=abstract or "",
        )
        with (text_dir / f"node_{idx}.txt").open("w", encoding="utf-8") as handle:
            handle.write(content)


def save_outputs(
    graph: nx.Graph,
    labels: List[int],
    splits: Dict[str, np.ndarray],
    processed_dir: Path,
    text_dir: Path,
    pkl_path: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed_dir / f"{DATASET_NAME}_graph.gpickle"
    labels_path = processed_dir / f"{DATASET_NAME}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"
    num_classes_path = processed_dir / "num_classes.txt"

    nx.write_gpickle(graph, graph_path)
    pd.Series(labels, dtype=int).to_csv(labels_path, index=False, header=False)
    np.savez_compressed(splits_path, **splits)
    with num_classes_path.open("w", encoding="utf-8") as handle:
        handle.write(str(len(set(labels))))

    # store for multimodal loader compatibility
    with pkl_path.open("wb") as handle:
        import pickle

        pickle.dump((graph, labels, DATASET_NAME), handle)


def main() -> None:
    ensure_pyarrow_available()
    args = parse_args()

    raw_dir = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} does not exist.")

    train_path = discover_split(args.train, raw_dir, "train*.parquet")
    val_path = discover_split(args.val, raw_dir, "val*.parquet")
    test_path = discover_split(args.test, raw_dir, "test*.parquet")

    print(f"Loading train split from {train_path}")
    df_train = parse_compound_column(pd.read_parquet(train_path))
    print(f"Loading validation split from {val_path}")
    df_val = parse_compound_column(pd.read_parquet(val_path))
    print(f"Loading test split from {test_path}")
    df_test = parse_compound_column(pd.read_parquet(test_path))

    for df, split_name in ((df_train, "train"), (df_val, "validation"), (df_test, "test")):
        df["__split__"] = split_name

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    # Normalise column names to lower case for easier access
    df_all.columns = [str(col).lower() for col in df_all.columns]

    required_columns = {"instruction", "answer"}
    if not required_columns.issubset(set(df_all.columns)):
        missing = required_columns - set(df_all.columns)
        raise KeyError(f"Missing required columns in parquet data: {sorted(missing)}")

    abstracts_col = "abstracts"
    if abstracts_col not in df_all.columns:
        # Some conversions use an alternative key such as 'contexts'
        for candidate in ("contexts", "passages", "support", "evidence"):
            if candidate in df_all.columns:
                abstracts_col = candidate
                break

    df_all["abstracts_text"] = df_all.get(abstracts_col, "").apply(flatten_abstracts)
    df_all["instruction"] = df_all["instruction"].fillna("").astype(str)
    df_all["answer"] = df_all["answer"].fillna("").astype(str)

    df_all["label"] = df_all["answer"].apply(infer_label)
    before = len(df_all)
    df_all = df_all.dropna(subset=["label"]).reset_index(drop=True)
    after = len(df_all)
    if after < before:
        print(f"Filtered out {before - after} samples with unsupported answer labels.")
    if after == 0:
        raise RuntimeError("No samples with valid labels were found. Check the answer formatting.")

    feature_text = (
        df_all["instruction"].str.strip()
        + " \n"
        + df_all["abstracts_text"].str.strip()
    )
    features, vectorizer = build_feature_matrix(feature_text.tolist(), args.max_features)
    print(f"Built TF-IDF features with shape {features.shape} (vocab size {len(vectorizer.vocabulary_)})")

    graph = build_graph(features, args.neighbors, args.min_similarity)
    print(f"Graph summary: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # enrich nodes with metadata
    for idx, row in df_all.iterrows():
        graph.nodes[idx]["split"] = row["__split__"]
        graph.nodes[idx]["instruction"] = row["instruction"]
        graph.nodes[idx]["answer_text"] = row["answer"]
        graph.nodes[idx]["label"] = int(row["label"])

    write_text_blobs(
        texts=df_all["instruction"].tolist(),
        answers=df_all["answer"].tolist(),
        abstracts=df_all["abstracts_text"].tolist(),
        text_dir=DEFAULT_TEXT_DIR if args.out_dir == Path("data") else args.out_dir / "texts" / DATASET_NAME,
    )

    split_indices: Dict[str, np.ndarray] = {}
    for name in df_all["__split__"].unique():
        mask = df_all.index[df_all["__split__"] == name].to_numpy(dtype=np.int64)
        key = "train" if name == "train" else ("val" if name.startswith("val") else name)
        split_indices[key] = mask
    # ensure test split exists
    if "test" not in split_indices:
        raise RuntimeError("Test split indices could not be determined.")

    if "val" not in split_indices and "validation" in split_indices:
        split_indices["val"] = split_indices.pop("validation")

    # Align directory targets with --out-dir
    processed_dir = DEFAULT_PROCESSED_DIR if args.out_dir == Path("data") else args.out_dir / DATASET_NAME / "processed"
    text_dir = DEFAULT_TEXT_DIR if args.out_dir == Path("data") else args.out_dir / "texts" / DATASET_NAME
    pkl_path = DEFAULT_PKL_PATH if args.out_dir == Path("data") else args.out_dir / f"{DATASET_NAME}.pkl"

    save_outputs(
        graph=graph,
        labels=df_all["label"].astype(int).tolist(),
        splits=split_indices,
        processed_dir=processed_dir,
        text_dir=text_dir,
        pkl_path=pkl_path,
    )

    print(f"Saved graph to {processed_dir}")
    print(f"Saved text descriptions to {text_dir}")
    print(f"Saved multimodal pickle to {pkl_path}")


if __name__ == "__main__":
    main()
