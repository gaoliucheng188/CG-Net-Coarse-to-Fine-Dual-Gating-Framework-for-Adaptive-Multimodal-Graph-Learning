from graphviz import Digraph

def create_framework_diagram():
    dot = Digraph(comment='Multimodal GNN Explainer Framework', format='png')
    dot.attr(rankdir='LR')
    dot.attr(splines='ortho')
    dot.attr(nodesep='0.6')
    dot.attr(ranksep='0.8')
    
    # Global node style
    dot.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='12')
    
    # Colors based on the user's image (approximate hex codes)
    # Blue section
    color_blue_bg = '#E6F3FF'
    color_blue_border = '#B3D9FF'
    color_blue_node = '#CCE5FF'
    
    # Orange section
    color_orange_bg = '#FFF5E6'
    color_orange_border = '#FFD9B3'
    color_orange_node = '#FFE5CC'
    
    # Purple section
    color_purple_bg = '#F2E6FF'
    color_purple_border = '#D9B3FF'
    color_purple_node = '#E5CCFF'

    # ---------------------------------------------------------
    # Cluster 1: Input & Preprocessing (Blue)
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='I. Input & Preprocessing Layer\n(Multimodal Data Input)', 
               style='filled', color=color_blue_bg, pencolor=color_blue_border, fontsize='16', fontcolor='#333333')
        
        c.node('graph_data', 'Graph Data\n(Adjacency A, Features X)', fillcolor=color_blue_node, color=color_blue_border)
        c.node('text_data', 'Text Data\n(Sequences/Documents)', fillcolor=color_blue_node, color=color_blue_border)
        c.node('image_data', 'Image Data\n(Pixel Grids)', fillcolor=color_blue_node, color=color_blue_border)
        
        c.node('preprocess', 'Preprocessing & Alignment\n(Tokenization, Normalization)', fillcolor='#FFFFFF', color=color_blue_border, shape='component')
        
        # Edges within cluster
        c.edge('graph_data', 'preprocess')
        c.edge('text_data', 'preprocess')
        c.edge('image_data', 'preprocess')

    # ---------------------------------------------------------
    # Cluster 2: Cognitive Feature Extraction (Orange)
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='II. Multimodal Feature Extraction Layer\n(Tri-stream Analysis)', 
               style='filled', color=color_orange_bg, pencolor=color_orange_border, fontsize='16', fontcolor='#333333')
        
        # Stream 1: Graph
        c.node('gcn', 'Graph Encoder\n(GCN/GAT Layers)', fillcolor=color_orange_node, color=color_orange_border)
        
        # Stream 2: Text
        c.node('text_enc', 'Text Encoder\n(BERT/LSTM)', fillcolor=color_orange_node, color=color_orange_border)
        
        # Stream 3: Image
        c.node('img_enc', 'Image Encoder\n(CNN/ResNet)', fillcolor=color_orange_node, color=color_orange_border)
        
        # Fusion
        c.node('fusion', 'Multimodal Fusion\n(Attention / Concat)', fillcolor='#FFCC99', color='#FFB366', shape='Mrecord')
        
        # Prediction
        c.node('pred', 'Prediction Head\n(Softmax)', fillcolor='#FFFFFF', color=color_orange_border)

        # Edges within cluster (will be connected from outside later)
        c.edge('gcn', 'fusion')
        c.edge('text_enc', 'fusion')
        c.edge('img_enc', 'fusion')
        c.edge('fusion', 'pred')

    # ---------------------------------------------------------
    # Cluster 3: Cognitive Prior Output Layer (Purple)
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='III. Explanation & Output Layer\n(GNNExplainer & Visualization)', 
               style='filled', color=color_purple_bg, pencolor=color_purple_border, fontsize='16', fontcolor='#333333')
        
        c.node('explainer', 'GNN Explainer\n(Mask Optimization)', fillcolor=color_purple_node, color=color_purple_border, shape='diamond')
        
        c.node('subgraph', 'Important Subgraph\n(Structure Explanation)', fillcolor='#FFFFFF', color=color_purple_border)
        c.node('features', 'Feature Importance\n(Text/Image/Node Feats)', fillcolor='#FFFFFF', color=color_purple_border)
        
        c.node('viz', 'Visualization\n(Heatmaps, Graphs)', fillcolor=color_purple_node, color=color_purple_border)
        
        c.edge('explainer', 'subgraph')
        c.edge('explainer', 'features')
        c.edge('subgraph', 'viz')
        c.edge('features', 'viz')

    # ---------------------------------------------------------
    # Edges between clusters
    # ---------------------------------------------------------
    dot.edge('preprocess', 'gcn', label='Graph Stream')
    dot.edge('preprocess', 'text_enc', label='Text Stream')
    dot.edge('preprocess', 'img_enc', label='Image Stream')
    
    dot.edge('pred', 'explainer', label='Backprop / Masking')

    # Save
    output_path = 'framework_diagram'
    dot.render(output_path, view=False)
    print(f"Diagram generated at {output_path}.png")

if __name__ == '__main__':
    create_framework_diagram()
