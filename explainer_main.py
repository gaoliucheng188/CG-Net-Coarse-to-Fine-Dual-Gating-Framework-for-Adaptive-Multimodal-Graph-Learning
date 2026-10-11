""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
import sys

import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain

from data.multimodal_loader import MultimodalDataset
from explainer.explain_multimodal import MultimodalExplainer
from models_multimodal import MultimodalGcnEncoderNode, MultimodalGcnEncoderGraph

def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--explainer-uma",
        dest="explainer_uma",
        action="store_true",
        help="Whether to use UMA (Attention) in the explainer module even if model is concat.",
    )
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )
    
    # 多模态参数
    parser.add_argument(
        "--fusion-method", dest="fusion_method", type=str, default="concat",
        help="Multimodal fusion method: concat, bilinear, attention"
    )
    parser.add_argument(
        "--multimodal", action="store_true", help="Whether to use multimodal features"
    )
    parser.add_argument(
        "--text-dim", dest="text_dim", type=int, default=300,
        help="Dimension of text features"
    )
    parser.add_argument(
        "--image-dim", dest="image_dim", type=int, default=2048,
        help="Dimension of image features"
    )
    parser.add_argument(
        "--force-text-fallback",
        dest="force_text_fallback",
        action="store_true",
        default=False,
        help="Force using tokenizer/hashing text embedding fallback instead of loading transformers models",
    )
    parser.add_argument(
        "--disable-text",
        dest="disable_text",
        action="store_true",
        default=False,
        help="Disable text modality when running multimodal explanations",
    )
    parser.add_argument(
        "--disable-image",
        dest="disable_image",
        action="store_true",
        default=False,
        help="Disable image modality when running multimodal explanations",
    )
    
    # Ablation Study Arguments
    parser.add_argument(
        "--ablation-no-attention",
        dest="ablation_no_attention",
        action="store_true",
        default=False,
        help="Ablation B1: Disable Unified Attention Layer (Dynamic Weights)",
    )
    parser.add_argument(
        "--ablation-no-interaction",
        dest="ablation_no_interaction",
        action="store_true",
        default=False,
        help="Ablation B2: Disable Interaction Detection Head",
    )
    parser.add_argument(
        "--ablation-no-domain-rules",
        dest="ablation_no_domain_rules",
        action="store_true",
        default=False,
        help="Ablation B3: Disable Domain Rule Base",
    )

    # Hyperparameters for Multimodal Explainer
    parser.add_argument("--lambda-ID", dest="lambda_ID", type=float, default=0.1, help="Coefficient for Interaction Loss")
    parser.add_argument("--lambda-consist", dest="lambda_consist", type=float, default=0.1, help="Coefficient for Consistency Loss")
    parser.add_argument("--lambda-domain", dest="lambda_domain", type=float, default=0.1, help="Coefficient for Domain Loss (Total)")
    parser.add_argument("--lambda-conn", dest="lambda_conn", type=float, default=0.1, help="Coefficient for Connectivity Loss (part of Domain Loss)")
    parser.add_argument("--lambda-valid", dest="lambda_valid", type=float, default=0.1, help="Coefficient for Validity Loss (part of Domain Loss)")

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",  
        opt_scheduler="none",
        cuda="1",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
        disable_text=False,
        disable_image=False,
    )
    return parser.parse_args()


def main():
    # Load a configuration
    prog_args = arg_parse()

    # 加载多模态数据集
    if prog_args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name=prog_args.dataset,
            gpu=prog_args.gpu
        )
        prog_args.multimodal_loader = multimodal_loader
    else:
        multimodal_loader = None

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"] # get computation graph

    def _infer_last_dim(array, default):
        if array is None:
            return default
        shape = getattr(array, "shape", None)
        if not shape:
            return default
        return shape[-1] if len(shape) >= 1 else default

    feat_tensor = cg_dict.get("feat")
    pred_tensor = cg_dict.get("pred")
    input_dim = _infer_last_dim(feat_tensor, 0)
    num_classes = _infer_last_dim(pred_tensor, 1)
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    # 构建模型
    print("Method: ", prog_args.method)
    if graph_mode:
        if prog_args.multimodal:
            model = MultimodalGcnEncoderGraph(
                input_dim=input_dim,
                hidden_dim=prog_args.hidden_dim,
                embedding_dim=prog_args.output_dim,
                label_dim=num_classes,
                num_layers=prog_args.num_gc_layers,
                bn=prog_args.bn,
                args=prog_args,
                force_text_fallback=prog_args.force_text_fallback,
                text_dim=prog_args.text_dim,
                image_dim=prog_args.image_dim,
                fusion_method=prog_args.fusion_method,
                disable_text=prog_args.disable_text,
                disable_image=prog_args.disable_image,
            )
        else:
            model = models.GcnEncoderGraph(
                input_dim=input_dim,
                hidden_dim=prog_args.hidden_dim,
                embedding_dim=prog_args.output_dim,
                label_dim=num_classes,
                num_layers=prog_args.num_gc_layers,
                bn=prog_args.bn,
                args=prog_args,
            )
    else:
        if prog_args.multimodal:
            model = MultimodalGcnEncoderNode(
                input_dim=input_dim,
                hidden_dim=prog_args.hidden_dim,
                embedding_dim=prog_args.output_dim,
                label_dim=num_classes,
                num_layers=prog_args.num_gc_layers,
                bn=prog_args.bn,
                args=prog_args,
                force_text_fallback=prog_args.force_text_fallback,
                text_dim=prog_args.text_dim,
                image_dim=prog_args.image_dim,
                fusion_method=prog_args.fusion_method,
                disable_text=prog_args.disable_text,
                disable_image=prog_args.disable_image,
            )
        else:
            model = models.GcnEncoderNode(
                input_dim=input_dim,
                hidden_dim=prog_args.hidden_dim,
                embedding_dim=prog_args.output_dim,
                label_dim=num_classes,
                num_layers=prog_args.num_gc_layers,
                bn=prog_args.bn,
                args=prog_args,
            )
    if prog_args.gpu:
        model = model.cuda()

    # Handle dynamic fusion layer resizing (e.g. if image modality was dropped during training)
    if prog_args.multimodal and "fusion.fc.weight" in ckpt["model_state"]:
        ckpt_fc_shape = ckpt["model_state"]["fusion.fc.weight"].shape
        current_fc_shape = model.fusion.fc.weight.shape
        if ckpt_fc_shape != current_fc_shape:
            print(f"Adjusting fusion.fc shape from {current_fc_shape} to {ckpt_fc_shape} to match checkpoint.")
            model.fusion.fc = torch.nn.Linear(ckpt_fc_shape[1], ckpt_fc_shape[0])
            if prog_args.gpu:
                model.fusion.fc = model.fusion.fc.cuda()

    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"], strict=False) 

    # 创建解释器
    if prog_args.multimodal:
        explainer = MultimodalExplainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_mode=graph_mode,
        graph_idx=prog_args.graph_idx,
        use_uma=prog_args.explainer_uma,
    )
    else:
        explainer = explain.Explainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
        )

    # TODO: API should definitely be cleaner
    # Let's define exactly which modes we support 
    # We could even move each mode to a different method (even file)
    if prog_args.explain_node is not None:
        explainer.explain(prog_args.explain_node, unconstrained=False)
    elif graph_mode:
        if prog_args.multigraph_class >= 0:
            print(cg_dict["label"])
            # only run for graphs with label specified by multigraph_class
            labels = cg_dict["label"].numpy()
            graph_indices = []
            for i, l in enumerate(labels):
                if l == prog_args.multigraph_class:
                    graph_indices.append(i)
                if len(graph_indices) > 30:
                    break
            print(
                "Graph indices for label ",
                prog_args.multigraph_class,
                " : ",
                graph_indices,
            )
            explainer.explain_graphs(graph_indices=graph_indices)

        elif prog_args.graph_idx == -1:
            # just run for a customized set of indices
            explainer.explain_graphs(graph_indices=[1, 2, 3, 4])
        else:
            explainer.explain(
                node_idx=0,
                graph_idx=prog_args.graph_idx,
                graph_mode=True,
                unconstrained=False,
            )
            io_utils.plot_cmap_tb(writer, "tab20", 20, "tab20_cmap")
    else:
        if prog_args.multinode_class >= 0:
            print(cg_dict["label"])
            # only run for nodes with label specified by multinode_class
            labels = cg_dict["label"][0]  # already numpy matrix

            node_indices = []
            for i, l in enumerate(labels):
                if len(node_indices) > 4:
                    break
                if l == prog_args.multinode_class:
                    node_indices.append(i)
            print(
                "Node indices for label ",
                prog_args.multinode_class,
                " : ",
                node_indices,
            )
            explainer.explain_nodes(node_indices, prog_args)

        else:
            # explain a set of nodes
            masked_adj = explainer.explain_nodes_gnn_stats(
                range(400, 700, 5), prog_args
            )
        
if __name__ == "__main__":
    main()