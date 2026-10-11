"""gengraph.py

   Generating and manipulation the synthetic graphs needed for the paper's experiments.
   Added multimodal data (text and image) generation functionality.
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

# Set matplotlib backend to file writing
plt.switch_backend("agg")

import networkx as nx
import numpy as np
from tensorboardX import SummaryWriter

from utils import synthetic_structsim
from utils import featgen
import utils.io_utils as io_utils


####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def join_graph(G1, G2, n_pert_edges):
    """ Join two graphs along matching nodes, then perturb the resulting graph.
    Args:
        G1, G2: Networkx graphs to be joined.
        n_pert_edges: number of perturbed edges.
    Returns:
        A new graph, result of merging and perturbing G1 and G2.
    """
    assert n_pert_edges > 0
    F = nx.compose(G1, G2)
    edge_cnt = 0
    while edge_cnt < n_pert_edges:
        node_1 = np.random.choice(list(G1.nodes()))
        node_2 = np.random.choice(list(G2.nodes()))
        F.add_edge(node_1, node_2)
        edge_cnt += 1
    return F


def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    # 新代码（兼容所有networkx版本）
    adj = np.array(nx.adjacency_matrix(G).todense())
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}


####################################
#
# Generating synthetic graphs
#
###################################
def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5, save_multimodal=True):
    """ Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)
        save_multimodal   :  Whether to generate and save multimodal data (text and images)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    # Generate and save multimodal data
    if save_multimodal:
        # Create save directories
        text_dir = os.path.join('data', 'texts', 'syn1')
        image_dir = os.path.join('data', 'images', 'syn1')
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Generate text descriptions and visualizations for each node
        for node in G.nodes():
            # Node-level text description
            node_role = role_id[list(G.nodes()).index(node)]  # Get role ID for current node
            neighbors = list(G.neighbors(node))
            text = (f"Node {node} with role {node_role}. "
                   f"Connected to {len(neighbors)} nodes: {neighbors[:5]}{'...' if len(neighbors) > 5 else ''}")
            with open(os.path.join(text_dir, f'node_{node}.txt'), 'w') as f:
                f.write(text)
            
            # Node-level visualization (subgraph of neighbors)
            subgraph_nodes = [node] + neighbors[:10]  # Limit to 10 neighbors for clarity
            subgraph = G.subgraph(subgraph_nodes)
            plt.figure(figsize=(4, 4))
            nx.draw(
                subgraph,
                with_labels=True,
                node_color=[role_id[list(G.nodes()).index(n)] for n in subgraph.nodes()],
                cmap=plt.cm.Set3
            )
            plt.title(f"Node {node} Neighborhood")
            plt.savefig(os.path.join(image_dir, f'node_{node}.png'), bbox_inches='tight')
            plt.close()
        
        # Generate graph-level text description
        graph_text = (f"Synthetic graph syn1 (name: {name}) with {G.number_of_nodes()} nodes "
                     f"and {G.number_of_edges()} edges. Contains {nb_shapes} house-shaped "
                     f"subgraphs attached to a Barabasi-Albert base graph.")
        with open(os.path.join(text_dir, 'graph_description.txt'), 'w') as f:
            f.write(graph_text)
        
        # Generate graph-level visualization
        plt.figure(figsize=(8, 8))
        nx.draw(
            G,
            node_color=role_id,
            cmap=plt.cm.Set3,
            node_size=50,
            alpha=0.8
        )
        plt.title(f"Synthetic Graph syn1: {name}")
        plt.savefig(os.path.join(image_dir, 'full_graph.png'), bbox_inches='tight')
        plt.close()

    return G, role_id, name


def gen_syn2(nb_shapes=100, width_basis=350, save_multimodal=True):
    """ Synthetic Graph #2:

    Start with Barabasi-Albert graph and add node features indicative of a community label.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        save_multimodal   :  Whether to generate and save multimodal data (text and images)

    Returns:
        G                 :  A networkx graph
        label             :  Label of the nodes (determined by role_id and community)
        name              :  A graph identifier
    """
    basis_type = "ba"

    random_mu = [0.0] * 8
    random_sigma = [1.0] * 8

    # Create two grids
    mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    feat_gen_G1 = featgen.GaussianFeatureGen(mu=mu_1, sigma=sigma_1)
    feat_gen_G2 = featgen.GaussianFeatureGen(mu=mu_2, sigma=sigma_2)
    G1, role_id1, name = gen_syn1(feature_generator=feat_gen_G1, m=4, save_multimodal=False)
    G2, role_id2, name = gen_syn1(feature_generator=feat_gen_G2, m=4, save_multimodal=False)
    G1_size = G1.number_of_nodes()
    num_roles = max(role_id1) + 1
    role_id2 = [r + num_roles for r in role_id2]
    label = role_id1 + role_id2

    # Edit node ids to avoid collisions on join
    g1_map = {n: i for i, n in enumerate(G1.nodes())}
    G1 = nx.relabel_nodes(G1, g1_map)
    g2_map = {n: i + G1_size for i, n in enumerate(G2.nodes())}
    G2 = nx.relabel_nodes(G2, g2_map)

    # Join
    n_pert_edges = width_basis
    G = join_graph(G1, G2, n_pert_edges)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes) + "_2comm"

    # Generate multimodal data for syn2
    if save_multimodal:
        text_dir = os.path.join('data', 'texts', 'syn2')
        image_dir = os.path.join('data', 'images', 'syn2')
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Graph-level description
        graph_text = (f"Synthetic graph syn2 (name: {name}) with {G.number_of_nodes()} nodes "
                     f"and {G.number_of_edges()} edges. Composed of two communities of Barabasi-Albert "
                     f"graphs with house subgraphs, connected by {n_pert_edges} perturbation edges.")
        with open(os.path.join(text_dir, 'graph_description.txt'), 'w') as f:
            f.write(graph_text)
        
        # Graph visualization with community coloring
        community_labels = [0]*G1_size + [1]*(G.number_of_nodes() - G1_size)
        plt.figure(figsize=(8, 8))
        nx.draw(
            G,
            node_color=community_labels,
            cmap=plt.cm.Set1,
            node_size=50,
            alpha=0.8
        )
        plt.title(f"Synthetic Graph syn2: {name} (Community View)")
        plt.savefig(os.path.join(image_dir, 'full_graph_community.png'), bbox_inches='tight')
        plt.close()

    return G, label, name


def gen_syn3(nb_shapes=80, width_basis=300, feature_generator=None, m=5, save_multimodal=True):
    """ Synthetic Graph #3:

    Start with Barabasi-Albert graph and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'grid') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)
        save_multimodal   :  Whether to generate and save multimodal data (text and images)

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["grid", 3]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    # Generate multimodal data for syn3
    if save_multimodal:
        text_dir = os.path.join('data', 'texts', 'syn3')
        image_dir = os.path.join('data', 'images', 'syn3')
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Graph-level description
        graph_text = (f"Synthetic graph syn3 (name: {name}) with {G.number_of_nodes()} nodes "
                     f"and {G.number_of_edges()} edges. Contains {nb_shapes} 3x3 grid subgraphs "
                     f"attached to a Barabasi-Albert base graph.")
        with open(os.path.join(text_dir, 'graph_description.txt'), 'w') as f:
            f.write(graph_text)
        
        # Graph visualization
        plt.figure(figsize=(8, 8))
        nx.draw(
            G,
            node_color=role_id,
            cmap=plt.cm.Set2,
            node_size=50,
            alpha=0.8
        )
        plt.title(f"Synthetic Graph syn3: {name}")
        plt.savefig(os.path.join(image_dir, 'full_graph.png'), bbox_inches='tight')
        plt.close()

    return G, role_id, name


def gen_syn4(nb_shapes=60, width_basis=8, feature_generator=None, m=4, save_multimodal=True):
    """ Synthetic Graph #4:

    Start with a tree and attach cycle-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'cycles') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'Tree').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.
        save_multimodal   :  Whether to generate and save multimodal data (text and images)

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["cycle", 6]] * nb_shapes

    fig = plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    # Generate multimodal data for syn4
    if save_multimodal:
        text_dir = os.path.join('data', 'texts', 'syn4')
        image_dir = os.path.join('data', 'images', 'syn4')
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Graph-level description
        graph_text = (f"Synthetic graph syn4 (name: {name}) with {G.number_of_nodes()} nodes "
                     f"and {G.number_of_edges()} edges. Contains {nb_shapes} 6-node cycle subgraphs "
                     f"attached to a tree base graph (depth: {m}).")
        with open(os.path.join(text_dir, 'graph_description.txt'), 'w') as f:
            f.write(graph_text)
        
        # Graph visualization
        plt.figure(figsize=(8, 8))
        nx.draw(
            G,
            node_color=role_id,
            cmap=plt.cm.Pastel1,
            node_size=50,
            alpha=0.8
        )
        plt.title(f"Synthetic Graph syn4: {name}")
        plt.savefig(os.path.join(image_dir, 'full_graph.png'), bbox_inches='tight')
        plt.close()

    path = os.path.join("log/syn4_base_h20_o20")
    writer = SummaryWriter(path)
    io_utils.log_graph(writer, G, "graph/full")

    return G, role_id, name


def gen_syn5(nb_shapes=80, width_basis=8, feature_generator=None, m=3, save_multimodal=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'grids') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'tree').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The grid dimension (m x m).
        save_multimodal   :  Whether to generate and save multimodal data (text and images)

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["grid", m]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0
    )
    G = perturb([G], 0.1)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    # Generate multimodal data for syn5
    if save_multimodal:
        text_dir = os.path.join('data', 'texts', 'syn5')
        image_dir = os.path.join('data', 'images', 'syn5')
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        
        # Graph-level description
        graph_text = (f"Synthetic graph syn5 (name: {name}) with {G.number_of_nodes()} nodes "
                     f"and {G.number_of_edges()} edges. Contains {nb_shapes} {m}x{m} grid subgraphs "
                     f"attached to a tree base graph.")
        with open(os.path.join(text_dir, 'graph_description.txt'), 'w') as f:
            f.write(graph_text)
        
        # Graph visualization
        plt.figure(figsize=(8, 8))
        nx.draw(
            G,
            node_color=role_id,
            cmap=plt.cm.Pastel2,
            node_size=50,
            alpha=0.8
        )
        plt.title(f"Synthetic Graph syn5: {name}")
        plt.savefig(os.path.join(image_dir, 'full_graph.png'), bbox_inches='tight')
        plt.close()

    path = os.path.join("log/syn5_base_h20_o20")
    writer = SummaryWriter(path)

    return G, role_id, name