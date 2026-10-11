#!/usr/bin/env python3
"""Preprocess the TUD-MUTAG benchmark and derive multimodal artefacts.

This utility reads the raw MUTAG files that follow the TUD graph kernel
convention (``MUTAG_A.txt`` etc.), builds a single disjoint-union graph with
node-level features, derives deterministic train/val/test node splits, and
emits the assets expected by ``MultimodalDataset``.

Generated outputs (under ``data/`` by default):
    - ``MUTAG.pkl``                          (pickled tuple (G, labels, name))
    - ``MUTAG/processed/mutag_graph.gpickle``
    - ``MUTAG/processed/mutag_labels.csv``
    - ``MUTAG/processed/split_masks.npz``
    - ``texts/MUTAG/node_*.txt``             (node-level text descriptions)
    - ``texts/MUTAG/graph_*.txt``            (graph-level summaries)
    - ``images/MUTAG/node_*.png``            (simple atom glyphs)

Example usage:
    python scripts/prepare_mutag.py --raw-dir data/tud_mutag/raw --out-dir data
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATASET_NAME = "MUTAG"
DEFAULT_RAW_DIR = Path("data") / "tud_mutag" / "raw"
DEFAULT_OUT_DIR = Path("data")
ATOM_SYMBOLS = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "I",
    5: "Cl",
    6: "Br",
}
BOND_LABELS = {
    0: "aromatic",
    1: "single",
    2: "double",
    3: "triple",
}
ATOM_COLORS = {
    "C": "#4a4a4a",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#3cc93c",
    "I": "#940094",
    "Cl": "#228822",
    "Br": "#a62929",
}
GRAPH_LABEL_NAMES = {0: "non-mutagenic", 1: "mutagenic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TUD-MUTAG with multimodal artefacts")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing MUTAG raw files (default: data/tud_mutag/raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Root output directory where processed assets are written (default: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of molecules assigned to the training split (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of molecules assigned to the validation split (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic graph-level splits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove previously generated text/image folders before writing new artefacts",
    )
    parser.add_argument(
        "--label-free-text",
        action="store_true",
        default=True,
        help="Write label-independent (clean) node/graph texts without class words (default: True)",
    )
    parser.add_argument(
        "--leaky-text",
        action="store_true",
        help="Use legacy label-leaking templates (overrides --label-free-text)",
    )
    return parser.parse_args()


def read_column_file(path: Path) -> List[int]:
    values: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            values.append(int(stripped))
    return values


def load_mutag_raw(raw_dir: Path) -> Dict[str, Sequence[int]]:
    required_files = [
        raw_dir / "MUTAG_A.txt",
        raw_dir / "MUTAG_edge_labels.txt",
        raw_dir / "MUTAG_graph_indicator.txt",
        raw_dir / "MUTAG_graph_labels.txt",
        raw_dir / "MUTAG_node_labels.txt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MUTAG raw files: " + ", ".join(missing)
        )

    edges: List[Tuple[int, int]] = []
    with (raw_dir / "MUTAG_A.txt").open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Malformed edge line: {raw_line!r}")
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            if u == v:
                continue
            edges.append((u, v))

    edge_labels = read_column_file(raw_dir / "MUTAG_edge_labels.txt")
    if len(edge_labels) != len(edges):
        raise ValueError(
            "Edge label file must match MUTAG_A.txt length. "
            f"Found {len(edge_labels)} labels for {len(edges)} adjacency entries."
        )

    graph_indicator = [value - 1 for value in read_column_file(raw_dir / "MUTAG_graph_indicator.txt")]
    graph_labels_raw = read_column_file(raw_dir / "MUTAG_graph_labels.txt")
    node_labels = read_column_file(raw_dir / "MUTAG_node_labels.txt")

    num_nodes = len(graph_indicator)
    if len(node_labels) != num_nodes:
        raise ValueError("Node label file size mismatch with graph indicator length")

    num_graphs = max(graph_indicator) + 1
    if len(graph_labels_raw) != num_graphs:
        raise ValueError(
            f"Graph labels length mismatch: expected {num_graphs} entries, got {len(graph_labels_raw)}"
        )

    # Normalize graph labels to {0, 1}
    unique_labels = sorted(set(graph_labels_raw))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    graph_labels = [label_map[val] for val in graph_labels_raw]

    return {
        "edges": edges,
        "edge_labels": edge_labels,
        "graph_indicator": graph_indicator,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "num_graphs": num_graphs,
        "num_nodes": num_nodes,
    }


def build_graph(payload: Dict[str, Sequence[int]]) -> nx.Graph:
    graph = nx.Graph()
    num_nodes = payload["num_nodes"]
    node_labels = payload["node_labels"]
    graph_indicator = payload["graph_indicator"]
    graph_labels = payload["graph_labels"]

    for node_id in range(num_nodes):
        atom_label = node_labels[node_id]
        graph_id = graph_indicator[node_id]
        graph.add_node(
            node_id,
            atom_label=atom_label,
            atom_symbol=ATOM_SYMBOLS.get(atom_label, f"X{atom_label}"),
            graph_id=graph_id,
            graph_label=graph_labels[graph_id],
        )

    # Deduplicate undirected edges and attach bond labels
    bond_lookup: Dict[Tuple[int, int], int] = {}
    for (u, v), label in zip(payload["edges"], payload["edge_labels"]):
        a, b = (u, v) if u < v else (v, u)
        if a == b:
            continue
        if (a, b) in bond_lookup and bond_lookup[(a, b)] != label:
            raise ValueError(
                f"Conflicting labels for edge {(a, b)}: {bond_lookup[(a, b)]} vs {label}"
            )
        bond_lookup.setdefault((a, b), label)

    for (u, v), label in bond_lookup.items():
        graph.add_edge(u, v, bond_label=label, bond_name=BOND_LABELS.get(label, "unknown"))

    # Derive node feature vectors (one-hot atom type + degree + bond histogram)
    atom_values = sorted({graph.nodes[n]["atom_label"] for n in graph.nodes})
    atom_to_idx = {value: idx for idx, value in enumerate(atom_values)}
    bond_values = sorted(BOND_LABELS.keys())
    max_degree = max((deg for _, deg in graph.degree()), default=1)
    for node_id in graph.nodes:
        atom_label = graph.nodes[node_id]["atom_label"]
        one_hot = np.zeros(len(atom_values), dtype=np.float32)
        one_hot[atom_to_idx[atom_label]] = 1.0
        degree = graph.degree(node_id)
        normalized_degree = degree / max_degree if max_degree else 0.0
        bond_counter = np.zeros(len(bond_values), dtype=np.float32)
        for neighbour in graph.neighbors(node_id):
            bond_label = graph.edges[node_id, neighbour]["bond_label"]
            if bond_label < len(bond_values):
                bond_counter[bond_label] += 1.0
        if bond_counter.sum() > 0:
            bond_counter = bond_counter / bond_counter.sum()
        feature_vector = np.concatenate(
            [one_hot, np.array([normalized_degree], dtype=np.float32), bond_counter],
        ).astype(np.float32)
        graph.nodes[node_id]["feat"] = feature_vector

    return graph


def derive_splits(
    graph_indicator: Sequence[int],
    graph_labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 to leave room for test split")

    num_graphs = max(graph_indicator) + 1
    rng = np.random.default_rng(seed)
    graph_indices = np.arange(num_graphs)
    rng.shuffle(graph_indices)

    train_cutoff = int(math.floor(train_ratio * num_graphs))
    val_cutoff = int(math.floor((train_ratio + val_ratio) * num_graphs))
    if train_cutoff == 0 or val_cutoff == train_cutoff:
        raise ValueError("train/val ratios produced empty splits; adjust ratios")
    train_graphs = set(graph_indices[:train_cutoff].tolist())
    val_graphs = set(graph_indices[train_cutoff:val_cutoff].tolist())
    test_graphs = set(graph_indices[val_cutoff:].tolist())
    if not test_graphs:
        # ensure at least one graph in test set
        test_graphs = {graph_indices[-1]}
        if graph_indices[-1] in val_graphs:
            val_graphs.remove(graph_indices[-1])
            if not val_graphs and len(graph_indices) > 2:
                val_graphs.add(graph_indices[-2])

    node_to_graph = list(graph_indicator)
    train_nodes = [idx for idx, gid in enumerate(node_to_graph) if gid in train_graphs]
    val_nodes = [idx for idx, gid in enumerate(node_to_graph) if gid in val_graphs]
    test_nodes = [idx for idx, gid in enumerate(node_to_graph) if gid in test_graphs]

    if not train_nodes or not test_nodes:
        raise RuntimeError("Derived node splits must contain at least one element")

    return {
        "train": np.array(sorted(train_nodes), dtype=np.int64),
        "val": np.array(sorted(val_nodes), dtype=np.int64),
        "test": np.array(sorted(test_nodes), dtype=np.int64),
    }


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
    path.mkdir(parents=True, exist_ok=True)


def generate_node_text(
    graph: nx.Graph,
    node_id: int,
    label_free: bool = True,
) -> str:
    """Generate node text.

    label_free=True (default, clean-text protocol): keep only topological
    descriptors (atom symbol, degree, neighbour-bond descriptions). No class
    words such as "mutagenic" / "non-mutagenic".

    label_free=False: legacy leaky template that embeds the molecule class
    name. Kept only for reproduction of the original template experiments.
    """
    atom_symbol = graph.nodes[node_id]["atom_symbol"]
    graph_id = graph.nodes[node_id]["graph_id"]
    degree = graph.degree(node_id)
    neighbour_descriptions: List[str] = []
    for neighbour in sorted(graph.neighbors(node_id)):
        bond_name = graph.edges[node_id, neighbour].get("bond_name", "bond")
        neighbour_symbol = graph.nodes[neighbour]["atom_symbol"]
        neighbour_descriptions.append(f"{bond_name} bond to {neighbour_symbol}")
    if neighbour_descriptions:
        neighbour_text = "; ".join(neighbour_descriptions)
        degree_text = f"It has degree {degree} with {neighbour_text}."
    else:
        degree_text = "It has no recorded bonds."

    if label_free:
        return (
            f"Atom {atom_symbol} (node {node_id}) inside molecule {graph_id}. "
            f"{degree_text}"
        )

    # Legacy leaky template (NOT for fair evaluation).
    graph_label = int(graph.nodes[node_id].get("graph_label", -1))
    label_name = GRAPH_LABEL_NAMES.get(graph_label, "unknown")
    return (
        f"Atom {atom_symbol} (node {node_id}) inside molecule {graph_id} "
        f"is classified as {label_name}. {degree_text}"
    )


def generate_graph_text(
    graph: nx.Graph,
    graph_id: int,
    node_indices: Sequence[int],
    graph_label: int,
    label_free: bool = True,
) -> str:
    atoms = Counter(graph.nodes[idx]["atom_symbol"] for idx in node_indices)
    lines = [
        f"Molecule {graph_id} summary:",
        f"Node count: {len(node_indices)}, edge count: {graph.subgraph(node_indices).number_of_edges()}.",
        "Atom composition: " + ", ".join(f"{symbol} x{count}" for symbol, count in sorted(atoms.items())),
    ]
    if not label_free:
        label_name = GRAPH_LABEL_NAMES.get(graph_label, "unknown")
        lines.insert(1, f"Class label: {label_name}.")
    return "\n".join(lines)


def persist_modalities(
    graph: nx.Graph,
    payload: Dict[str, Sequence[int]],
    text_dir: Path,
    image_dir: Path,
    overwrite: bool,
    label_free: bool = True,
) -> None:
    ensure_clean_dir(text_dir, overwrite)
    ensure_clean_dir(image_dir, overwrite)

    node_to_graph: Dict[int, List[int]] = defaultdict(list)
    for node_idx, graph_idx in enumerate(payload["graph_indicator"]):
        node_to_graph[graph_idx].append(node_idx)

    # Node-level text + glyphs
    for node_id in graph.nodes:
        text_path = text_dir / f"node_{node_id}.txt"
        text_path.write_text(
            generate_node_text(graph, node_id, label_free=label_free),
            encoding="utf-8",
        )

        colour = ATOM_COLORS.get(graph.nodes[node_id]["atom_symbol"], "#6c6c6c")
        bond_counter = Counter(
            graph.edges[node_id, neigh].get("bond_name", "bond")
            for neigh in graph.neighbors(node_id)
        )
        image_path = image_dir / f"node_{node_id}.png"
        draw_atom_glyph(
            symbol=graph.nodes[node_id]["atom_symbol"],
            colour=colour,
            bonds=bond_counter,
            out_path=image_path,
        )

    # Graph-level summaries (text) for compatibility with MultimodalDataset.get_graph_data
    for graph_id, node_indices in node_to_graph.items():
        text_path = text_dir / f"graph_{graph_id}.txt"
        text_path.write_text(
            generate_graph_text(
                graph,
                graph_id,
                node_indices,
                graph.nodes[node_indices[0]]["graph_label"],
                label_free=label_free,
            ),
            encoding="utf-8",
        )


def draw_atom_glyph(symbol: str, colour: str, bonds: Counter, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=120)
    ax.set_axis_off()
    circle = plt.Circle((0.5, 0.55), 0.45, color=colour, alpha=0.9)
    ax.add_patch(circle)
    ax.text(0.5, 0.55, symbol, ha="center", va="center", fontsize=24, color="white", fontweight="bold")
    if bonds:
        bond_text = "\n".join(f"{name[:3]}: {count}" for name, count in bonds.most_common(3))
        ax.text(0.5, 0.15, bond_text, ha="center", va="center", fontsize=10, color="white")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_artifacts(
    graph: nx.Graph,
    labels: Sequence[int],
    splits: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    processed_dir = out_dir / DATASET_NAME / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed_dir / "mutag_graph.gpickle"
    labels_path = processed_dir / "mutag_labels.csv"
    splits_path = processed_dir / "split_masks.npz"
    num_classes_path = processed_dir / "num_classes.txt"
    pkl_path = out_dir / f"{DATASET_NAME}.pkl"

    nx.write_gpickle(graph, graph_path)
    pd.Series(labels, dtype=int).to_csv(labels_path, index=False, header=False)
    np.savez_compressed(splits_path, **splits)
    num_classes_path.write_text(str(len(set(labels))), encoding="utf-8")

    with pkl_path.open("wb") as handle:
        import pickle
        pickle.dump((graph, list(labels), DATASET_NAME), handle)


def main() -> None:
    args = parse_args()

    raw_dir = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} does not exist")

    payload = load_mutag_raw(raw_dir)
    graph = build_graph(payload)
    labels = [graph.nodes[idx]["graph_label"] for idx in range(payload["num_nodes"])]
    splits = derive_splits(
        payload["graph_indicator"],
        payload["graph_labels"],
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    text_dir = args.out_dir / "texts" / DATASET_NAME
    image_dir = args.out_dir / "images" / DATASET_NAME
    label_free = not bool(getattr(args, "leaky_text", False))
    if getattr(args, "label_free_text", True) is False:
        label_free = False
    print(f"Text protocol: {'label-free (clean)' if label_free else 'legacy leaky template'}")
    persist_modalities(
        graph,
        payload,
        text_dir,
        image_dir,
        overwrite=args.overwrite,
        label_free=label_free,
    )
    save_artifacts(graph, labels, splits, args.out_dir)

    print("Finished preparing MUTAG dataset")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  - texts: {text_dir} ({'clean' if label_free else 'leaky'})")
    print(f"  - images: {image_dir}")
    print(f"  - processed: {args.out_dir / DATASET_NAME / 'processed'}")
    unique, counts = np.unique(labels, return_counts=True)
    stats = {GRAPH_LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)}
    print("Class distribution (nodes):", stats)
    print("Artifacts written to:")
    print(f"  - {args.out_dir / DATASET_NAME / 'processed'}")
    print(f"  - {text_dir}")
    print(f"  - {image_dir}")
    print(f"  - {args.out_dir / f'{DATASET_NAME}.pkl'}")


if __name__ == "__main__":
    main()

