import warnings
warnings.filterwarnings("ignore")
""" train.py

    Main interface to train the GNNs that will be later explained.
"""
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg' 等，根据系统安装的库选择
import matplotlib.pyplot as plt
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import argparse
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import pandas as pd

import torch
import torch.nn as nn

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import configs
from utils import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models
from models_multimodal import MultimodalGcnEncoderNode, MultimodalGcnEncoderGraph
from data.multimodal_loader import MultimodalDataset

import torch
# 添加这一行，将Variable映射为torch.tensor
Variable = torch.tensor

#############################
#
# Prepare Data
#
#############################
def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    print("Number of graphs: ", len(graphs))
    print("Number of edges: ", sum([G.number_of_edges() for G in graphs]))
    print(
        "Max, avg, std of graph size: ",
        max([G.number_of_nodes() for G in graphs]),
        ", " "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
        ", " "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])),
    )

    # minibatch
    dataset_sampler = graph_utils.GraphSampler(
        train_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        val_graphs, 
        normalize=False, 
        max_num_nodes=max_nodes, 
        features=args.feature_type
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        test_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        dataset_sampler.max_num_nodes,
        dataset_sampler.feat_dim,
        dataset_sampler.assign_feat_dim,
    )

#############################
#
# Training 
#
#############################
def train(
    dataset,
    model,
    args,
    multimodal_loader=None,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
):
    writer_batch_idx = [0, 3, 6, 9]

    # build optimizer and (optionally) scheduler from train_utils so args.lr/opt take effect
    scheduler, optimizer = train_utils.build_optimizer(
        args, filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)
            
            # 基础数据准备 -> 移动到统一 device
            device = args.device if hasattr(args, 'device') else torch.device('cpu')
            adj = Variable(data["adj"].float(), requires_grad=False).to(device)
            h0 = Variable(data["feats"].float(), requires_grad=False).to(device)
            label = Variable(data["label"].long()).to(device)
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).to(device)

            # 多模态数据处理
            text_feats = None
            image_feats = None
            if args.multimodal and multimodal_loader is not None:
                # 获取当前批次节点ID（假设节点ID在数据集中连续编号）
                start_idx = batch_idx * args.batch_size
                end_idx = start_idx + len(data["num_nodes"])
                node_ids = list(range(start_idx, end_idx))
                
                # 提取多模态特征
                text_list = []
                image_list = []
                for node_id in node_ids:
                    node_data = multimodal_loader.get_node_data(node_id)
                    text_list.append(node_data["text"])
                    image_list.append(node_data["image_path"])
                
                # 转换为张量并移动到 device
                text_feats = torch.stack(text_list).to(device)
                image_feats = torch.stack(image_list).to(device)

            # 前向传播（支持多模态输入）
            if args.multimodal:
                    ypred, att_adj = model(
                        h0, adj, batch_num_nodes, 
                        assign_x=assign_input,
                        text_feats=text_feats,
                        image_feats=image_feats
                    )
            else:
                    ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)

            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            # 损失计算
            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(dataset, model, args, multimodal_loader, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, multimodal_loader, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, multimodal_loader, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    # 注释掉非交互式后端，使用默认的交互式后端（如TkAgg）
    # plt.switch_backend("agg")  
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)

    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])

    # 先保存图片（可选）
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    # 显示图像
    plt.show()
    # 保持图像窗口打开，直到手动关闭
    plt.close()
    matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs


def train_node_classifier(G, labels, model, args, multimodal_loader=None, writer=None, splits=None):
    # 节点级训练/测试拆分
    num_nodes = G.number_of_nodes()

    val_idx = []
    if splits is not None:
        def _ensure_list(key, default=None):
            arr = splits.get(key) if splits is not None else None
            if arr is None:
                return [] if default is None else default
            if isinstance(arr, (list, tuple)):
                return [int(x) for x in arr]
            return [int(x) for x in np.asarray(arr).tolist()]

        train_idx = _ensure_list("train")
        val_idx = _ensure_list("val") or _ensure_list("validation")
        test_idx = _ensure_list("test")
        if not train_idx:
            raise ValueError("Provided splits must include a non-empty 'train' set")
        if not test_idx and not val_idx:
            raise ValueError("Provided splits must include at least one evaluation split ('val' or 'test')")
        if not test_idx:
            # 如果没有单独的测试集，就把验证集当成测试集
            test_idx = val_idx
        eval_idx = val_idx if val_idx else test_idx
    else:
        num_train = int(num_nodes * args.train_ratio)
        idx = [i for i in range(num_nodes)]
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
        eval_idx = test_idx

    data = gengraph.preprocess_input_graph(G, labels)
    # 修正labels为1D
    node_labels = data["labels"]
    if isinstance(node_labels, np.ndarray) and node_labels.ndim > 1:
        if node_labels.shape[0] == 1:
            node_labels = node_labels[0]
        else:
            node_labels = node_labels.squeeze()
    labels_train = torch.tensor(node_labels[train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        # 多模态特征处理
        text_feats = None
        image_feats = None
        if args.multimodal and multimodal_loader is not None:
            text_list = []
            image_list = []
            for node_id in range(num_nodes):
                node_data = multimodal_loader.get_node_data(node_id)
                text_list.append(node_data["text"])
                image_list.append(node_data["image_path"])
            text_feats = text_list  # 直接传字符串列表
            image_feats = image_list  # 直接传路径列表

        # 前向传播 -> 使用统一 device
        device = args.device if hasattr(args, 'device') else torch.device('cpu')
        x_dev = x.to(device)
        adj_dev = adj.to(device)
        if args.multimodal:
            # text_feats and image_feats are lists or tensors already handled
            ypred, adj_att = model(x_dev, adj_dev, text_feats=text_feats, image_feats=image_feats)
        else:
            ypred, adj_att = model(x_dev, adj_dev)
        # 损失计算：尽量鲁棒地从模型输出 ypred 中抽取训练节点对应的预测
        # 可能的 ypred 形状有：
        # - [num_nodes, num_classes]
        # - [num_classes, num_nodes]
        # - [num_nodes] (二分类/单值情况)
        # 使用已知的 num_nodes 判断哪一维是节点维度
        ypred_train = None
        try:
            # num_nodes 在函数开头已定义
            if ypred.dim() == 1:
                # 一维情况：直接按索引取
                ypred_train = ypred[train_idx]
            elif ypred.dim() == 2:
                # 判断哪个维度对应节点数量
                if ypred.shape[0] == num_nodes:
                    # 第一维是节点
                    ypred_train = ypred[train_idx]
                elif ypred.shape[1] == num_nodes:
                    # 第二维是节点
                    # 将列取出，转置为 [num_train, num_classes]
                    ypred_train = ypred[:, train_idx].T
                else:
                    # 退而求其次：如果任一维度覆盖最大的 train_idx，则尝试对应取值
                    max_idx = max(train_idx)
                    if ypred.shape[0] > max_idx:
                        ypred_train = ypred[train_idx]
                    elif ypred.shape[1] > max_idx:
                        ypred_train = ypred[:, train_idx].T
            else:
                # 支持更高维度（batch, nodes, classes）情况：尝试识别节点维度
                # 常见为 [batch, num_nodes, num_classes]
                if ypred.dim() == 3:
                    # 找到等于 num_nodes 的维度
                    node_dim = None
                    for d in range(ypred.dim()):
                        if ypred.shape[d] == num_nodes:
                            node_dim = d
                            break
                    if node_dim is not None:
                        # 假设最后一维是 classes 且节点维在中间
                        if node_dim == 1:
                            # batch x nodes x classes -> 拼接 batch 维后取 train_idx
                            b, n, c = ypred.shape
                            ypred_flat = ypred.reshape(b * n, c)
                            # train_idx 应指向单一图上的节点，这种情况不确定，直接尝试索引
                            ypred_train = ypred_flat[train_idx]
                        else:
                            # 未能处理的高维情况，交给后续异常捕获
                            ypred_train = None
        except Exception:
            ypred_train = None

        if ypred_train is None:
            raise ValueError(f"无法从模型输出 ypred 中提取训练节点预测 (ypred.shape={ypred.shape}, num_nodes={num_nodes}, train_idx sample={train_idx[:5]})")

        # labels_train应为1D类别索引
        if labels_train.dim() > 1:
            labels_train = labels_train.squeeze()

        device = args.device if hasattr(args, 'device') else torch.device('cpu')
        loss = model.loss(ypred_train, labels_train.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        elapsed = time.time() - begin_time
        if splits is not None:
            result_train, eval_result = evaluate_node(
                ypred.cpu(), data["labels"], train_idx, eval_idx
            )
            if val_idx and test_idx and eval_idx is val_idx:
                _, result_test = evaluate_node(
                    ypred.cpu(), data["labels"], train_idx, test_idx
                )
            else:
                result_test = eval_result
        else:
            result_train, result_test = evaluate_node(
                ypred.cpu(), data["labels"], train_idx, test_idx
            )
            eval_result = None
        
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            prec_payload = {"train": result_train["prec"], "test": result_test["prec"]}
            recall_payload = {"train": result_train["recall"], "test": result_test["recall"]}
            acc_payload = {"train": result_train["acc"], "test": result_test["acc"]}
            if splits is not None and val_idx:
                prec_payload["val"] = eval_result["prec"] if eval_result else result_test["prec"]
                recall_payload["val"] = eval_result["recall"] if eval_result else result_test["recall"]
                acc_payload["val"] = eval_result["acc"] if eval_result else result_test["acc"]
            writer.add_scalars("prec", prec_payload, epoch)
            writer.add_scalars("recall", recall_payload, epoch)
            writer.add_scalars("acc", acc_payload, epoch)

        if epoch % 10 == 0:
            if splits is not None and val_idx:
                print(
                    f"epoch: {epoch} | loss: {loss.item():.4f} | train_acc: {result_train['acc']:.4f} "
                    f"| val_acc: {eval_result['acc']:.4f} | test_acc: {result_test['acc']:.4f} | time: {elapsed:.2f}s"
                )
            else:
                print(
                    f"epoch: {epoch} | loss: {loss.item():.4f} | train_acc: {result_train['acc']:.4f} "
                    f"| test_acc: {result_test['acc']:.4f} | time: {elapsed:.2f}s"
                )

        if scheduler is not None:
            scheduler.step()
    

    # 保存计算图数据
    model.eval()
    device = args.device if hasattr(args, 'device') else torch.device('cpu')
    x_dev = x.to(device)
    adj_dev = adj.to(device)
    if args.multimodal:
        ypred, _ = model(x_dev, adj_dev, text_feats=text_feats, image_feats=image_feats)
    else:
        ypred, _ = model(x_dev, adj_dev)

    if splits is not None:
        train_final, eval_final = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, eval_idx
        )
        if val_idx and test_idx and eval_idx is val_idx:
            _, final_test = evaluate_node(ypred.cpu(), data["labels"], train_idx, test_idx)
        else:
            final_test = eval_final
        if val_idx:
            print(
                f"Final evaluation -> train_acc: {train_final['acc']:.4f}, val_acc: {eval_final['acc']:.4f}, test_acc: {final_test['acc']:.4f}"
            )
        else:
            print(
                f"Final evaluation -> train_acc: {train_final['acc']:.4f}, test_acc: {final_test['acc']:.4f}"
            )
    else:
        train_final, final_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        print(
            f"Final evaluation -> train_acc: {train_final['acc']:.4f}, test_acc: {final_test['acc']:.4f}"
        )
    
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": list(train_idx),
    }
    if splits is not None:
        cg_data["val_idx"] = list(val_idx)
        cg_data["test_idx"] = list(test_idx)
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)


def train_node_classifier_multigraph(G_list, labels, model, args, multimodal_loader=None, writer=None):
    train_idx_all, test_idx_all = [], []
    num_nodes = G_list[0].number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    train_idx_all.append(train_idx)
    test_idx = idx[num_train:]
    test_idx_all.append(test_idx)

    data = gengraph.preprocess_input_graph(G_list[0], labels[0])
    all_labels = data["labels"]
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)

    for i in range(1, len(G_list)):
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        train_idx_all.append(train_idx)
        test_idx = idx[num_train:]
        test_idx_all.append(test_idx)
        data = gengraph.preprocess_input_graph(G_list[i], labels[i])
        all_labels = np.concatenate((all_labels, data["labels"]), axis=0)
        labels_train = torch.cat(
            [
                labels_train,
                torch.tensor(data["labels"][:, train_idx], dtype=torch.long),
            ],
            dim=0,
        )
        adj = torch.cat([adj, torch.tensor(data["adj"], dtype=torch.float)])
        x = torch.cat(
            [x, torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)]
        )

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        # 多模态特征处理
        text_feats = None
        image_feats = None
        if args.multimodal and multimodal_loader is not None:
            text_list = []
            image_list = []
            total_nodes = x.shape[0] // len(G_list)
            for graph_id in range(len(G_list)):
                for node_id in range(total_nodes):
                    full_id = graph_id * total_nodes + node_id
                    node_data = multimodal_loader.get_node_data(full_id)
                    text_list.append(node_data["text"])
                    image_list.append(node_data["image"])
            text_feats = torch.stack(text_list)
            image_feats = torch.stack(image_list)
            device = args.device if hasattr(args, 'device') else torch.device('cpu')
            text_feats = text_feats.to(device)
            image_feats = image_feats.to(device)

        # 前向传播 -> 使用统一 device
        device = args.device if hasattr(args, 'device') else torch.device('cpu')
        x_dev = x.to(device)
        adj_dev = adj.to(device)
        if args.multimodal:
            ypred = model(x_dev, adj_dev, text_feats=text_feats, image_feats=image_feats)
        else:
            ypred = model(x_dev, adj_dev)

        # 损失计算
        ypred_train_cmp = torch.cat(
            [ypred[i, train_idx_all[i], :] for i in range(len(G_list))], dim=0
        ).reshape(len(G_list), num_train, -1)
        device = args.device if hasattr(args, 'device') else torch.device('cpu')
        loss = model.loss(ypred_train_cmp, labels_train.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        elapsed = time.time() - begin_time
        result_train, result_test = evaluate_node(
            ypred.cpu(), all_labels, train_idx_all, test_idx_all
        )
        
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        print(
            "epoch: ", epoch, 
            "; loss: ", loss.item(), 
            "; train_acc: ", result_train["acc"], 
            "; test_acc: ", result_test["acc"], 
            "; epoch time: ", "{0:0.2f}".format(elapsed)
        )

        if scheduler is not None:
            scheduler.step()
    
    print("Train confusion matrix:\n", result_train["conf_mat"])
    print("Test confusion matrix:\n", result_test["conf_mat"])

    # 保存计算图数据
    model.eval()
    device = args.device if hasattr(args, 'device') else torch.device('cpu')
    x_dev = x.to(device)
    adj_dev = adj.to(device)
    if args.multimodal:
        ypred = model(x_dev, adj_dev, text_feats=text_feats, image_feats=image_feats)
    else:
        ypred = model(x_dev, adj_dev)
    
    cg_data = {
        "adj": adj.cpu().detach().numpy(),
        "feat": x.cpu().detach().numpy(),
        "label": all_labels,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx_all,
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)


#############################
#
# Evaluate Trained Model
#
#############################
def evaluate(dataset, model, args, multimodal_loader=None, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    device = args.device if hasattr(args, 'device') else torch.device('cpu')
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).to(device)
        h0 = Variable(data["feats"].float()).to(device)
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).to(device)

        # 多模态特征处理
        text_feats = None
        image_feats = None
        if args.multimodal and multimodal_loader is not None:
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + len(data["num_nodes"])
            node_ids = list(range(start_idx, end_idx))
            
            text_list = []
            image_list = []
            for node_id in node_ids:
                node_data = multimodal_loader.get_node_data(node_id)
                text_list.append(node_data["text"])
                image_list.append(node_data["image_path"])
            
            text_feats = torch.stack(text_list).to(device)
            image_feats = torch.stack(image_list).to(device)

        # 前向传播
        if args.multimodal:
            ypred, att_adj = model(
                h0, adj, batch_num_nodes, 
                assign_x=assign_input,
                text_feats=text_feats,
                image_feats=image_feats
            )
        else:
            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_node(ypred, labels, train_idx, test_idx):
    # Robustly compute predicted labels per node for different ypred shapes
    # ypred can be: [num_nodes, num_classes], [batch, num_nodes, num_classes], [num_classes, num_nodes], etc.
    if isinstance(ypred, np.ndarray):
        ypred_t = torch.from_numpy(ypred)
    else:
        ypred_t = ypred

    if ypred_t.dim() == 3:
        # assume [batch, num_nodes, num_classes]
        pred_labels = torch.argmax(ypred_t, dim=2).cpu().numpy()
        # if batch size == 1, squeeze to [num_nodes]
        if pred_labels.shape[0] == 1:
            pred_labels = pred_labels[0]
    elif ypred_t.dim() == 2:
        # assume [num_nodes, num_classes] or [batch, num_classes]
        # pick argmax over last dim
        pred_labels = torch.argmax(ypred_t, dim=1).cpu().numpy()
    else:
        # fall back: flatten then argmax
        pred_labels = torch.argmax(ypred_t.view(-1, ypred_t.size(-1)), dim=1).cpu().numpy()

    # 处理单图和多图情况
    if isinstance(train_idx, list) and isinstance(train_idx[0], list):
        # 多图情况
        pred_train = np.concatenate([
            np.ravel(pred_labels[i, train_idx[i]]) 
            for i in range(len(train_idx))
        ])
        pred_test = np.concatenate([
            np.ravel(pred_labels[i, test_idx[i]]) 
            for i in range(len(test_idx))
        ])
        labels_train = np.concatenate([
            np.ravel(labels[i, train_idx[i]]) 
            for i in range(len(train_idx))
        ])
        labels_test = np.concatenate([
            np.ravel(labels[i, test_idx[i]]) 
            for i in range(len(test_idx))
        ])
    else:
        # 单图情况
        # 修正labels为1D
        if isinstance(labels, np.ndarray) and labels.ndim > 1:
            if labels.shape[0] == 1:
                labels = labels[0]
            else:
                labels = labels.squeeze()
        pred_train = np.ravel(pred_labels[train_idx])
        pred_test = np.ravel(pred_labels[test_idx])
        labels_train = np.ravel(labels[train_idx])
        labels_test = np.ravel(labels[test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test


#############################
#
# Run Experiments
#
#############################
def ppi_essential_task(args, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="ppi_essential",
            gpu=args.gpu
        )

    # 加载数据
    G = io_utils.read_biosnap(
        "data/ppi_essential",
        "hi-union-ppi.tsv",
        "G-HumanEssential.tsv",
        feat_file="G-MtfPathways_gene-motifs.csv",
    )
    labels = np.array([G.nodes[u]["label"] for u in G.nodes()])
    num_classes = max(labels) + 1
    input_dim = G.nodes[next(iter(G.nodes()))]["feat"].shape[0]

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
                bn=args.bn,
                args=args,
                force_text_fallback=args.force_text_fallback,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method
        )
    else:
        if args.method == "attn":
            print("Method: attn")
            model = models.GcnEncoderNode(
                input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )
        else:
            print("Method:", args.method)
            device = args.device if hasattr(args, 'device') else torch.device('cpu')
            args.loss_weight = torch.tensor([1, 5.0], dtype=torch.float).to(device)
            model = models.GcnEncoderNode(
                input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )

    # 将 model 移动到统一 device
    if hasattr(args, 'device'):
        model = model.to(args.device)

    train_node_classifier(G, labels, model, args, multimodal_loader, writer=writer)


def syn_task1(args, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="syn1",
            gpu=args.gpu
        )

    # 加载数据
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            force_text_fallback=args.force_text_fallback,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method
        )
    else:
        if args.method == "att":
            print("Method: att")
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )
        else:
            print("Method:", args.method)
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )

    if hasattr(args, 'device'):
        model = model.to(args.device)

    train_node_classifier(G, labels, model, args, multimodal_loader, writer=writer)


def syn_task2(args, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="syn2",
            gpu=args.gpu
        )

    # 加载数据
    G, labels, name = gengraph.gen_syn2()
    input_dim = len(G.nodes[0]["feat"])
    num_classes = max(labels) + 1

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            force_text_fallback=args.force_text_fallback,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method
        )
    else:
        if args.method == "attn":
            print("Method: attn")
            model = models.GcnEncoderNode(
                input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )
        else:
            print("Method:", args.method)
            model = models.GcnEncoderNode(
                input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )

    if hasattr(args, 'device'):
        model = model.to(args.device)

    train_node_classifier(G, labels, model, args, multimodal_loader, writer=writer)


def syn_task3(args, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="syn3",
            gpu=args.gpu
        )

    # 加载数据
    G, labels, name = gengraph.gen_syn3(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method
        )
    else:
        if args.method == "attn":
            print("Method: attn")
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )
        else:
            print("Method:", args.method)
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )

    if hasattr(args, 'device'):
        model = model.to(args.device)

    train_node_classifier(G, labels, model, args, multimodal_loader, writer=writer)


def syn_task4(args, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="syn4",
            gpu=args.gpu
        )

    # 加载数据
    G, labels, name = gengraph.gen_syn4(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method
        )
    else:
        if args.method == "attn":
            print("Method: attn")
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )
        else:
            print("Method:", args.method)
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )

    if hasattr(args, 'device'):
        model = model.to(args.device)

    train_node_classifier(G, labels, model, args, multimodal_loader, writer=writer)


def syn_task5(args, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="syn5",
            gpu=args.gpu
        )

    # 加载数据
    G, labels, name = gengraph.gen_syn5(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method
        )
    else:
        if args.method == "attn":
            print("Method: attn")
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )
        else:
            print("Method: base")
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, num_classes,
                args.num_gc_layers, bn=args.bn, args=args
            )

    if hasattr(args, 'device'):
        model = model.to(args.device)
    
def pqaref_task(args, writer=None):
    dataset_name = "pqaref"
    processed_dir = Path("data") / dataset_name / "processed"
    graph_path = processed_dir / f"{dataset_name}_graph.gpickle"
    labels_path = processed_dir / f"{dataset_name}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"
    fallback_pkl = Path("data") / f"{dataset_name}.pkl"

    if graph_path.exists() and labels_path.exists():
        G = nx.read_gpickle(graph_path)
        labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    elif fallback_pkl.exists():
        with open(fallback_pkl, "rb") as fh:
            G, labels_list, _ = pickle.load(fh)
            labels_list = list(map(int, labels_list))
        print("Warning: falling back to pickle load for PQAref. Run scripts/prepare_pqaref.py for the latest artefacts.")
    else:
        raise FileNotFoundError(
            "Processed PQAref artefacts not found. Please run 'python scripts/prepare_pqaref.py' first."
        )

    splits = None
    if splits_path.exists():
        npz = np.load(splits_path)
        splits = {key: npz[key] for key in npz.files}
    else:
        print("Warning: split_masks.npz not found for PQAref; using random data split during training.")

    labels_np = np.asarray(labels_list, dtype=int)
    if labels_np.ndim != 1:
        labels_np = labels_np.reshape(-1)
    num_classes = int(labels_np.max()) + 1 if labels_np.size > 0 else 0
    if num_classes <= 1:
        raise RuntimeError("PQAref preprocessing produced <=1 class, cannot train classifier.")

    input_dim = G.nodes[list(G.nodes())[0]]["feat"].shape[0]
    print(
        f"Loaded PQAref dataset: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {num_classes} classes "
        f"(feature_dim={input_dim})"
    )

    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name=dataset_name,
            gpu=args.gpu,
        )

    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method,
            force_text_fallback=args.force_text_fallback,
            disable_text=args.disable_text,
            disable_image=args.disable_image,
        )
    else:
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

    if hasattr(args, "device"):
        model = model.to(args.device)

    # Compute optional class weights to mitigate imbalance
    try:
        total_samples = labels_np.shape[0]
        unique, counts = np.unique(labels_np, return_counts=True)
        counts_full = np.zeros(num_classes, dtype=np.int64)
        counts_full[unique.astype(int)] = counts.astype(np.int64)
        eps = 1e-6
        scheme = getattr(args, "weight_scheme", "balanced")
        if scheme == "balanced":
            weights = total_samples / (float(num_classes) * (counts_full + eps))
        elif scheme == "inv_sqrt":
            weights = 1.0 / (np.sqrt(counts_full + eps))
            weights = weights * (total_samples / (float(num_classes) * np.mean(weights)))
        elif scheme == "log":
            weights = 1.0 / (np.log1p(counts_full) + eps)
            weights = weights * (total_samples / (float(num_classes) * np.mean(weights)))
        else:
            weights = np.ones(num_classes, dtype=float)
        max_w = getattr(args, "max_class_weight", 50.0)
        weights = np.minimum(weights, max_w)
        weights[counts_full == 0] = 0.0
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        weights_tensor = weights_tensor.to(args.device if hasattr(args, "device") else torch.device("cpu"))

        if getattr(args, "use_focal", False):
            gamma = getattr(args, "focal_gamma", 2.0)
            model.celoss = models.FocalLoss(gamma=gamma, weight=weights_tensor)
            print(f"Using FocalLoss (gamma={gamma}) with computed class weights for PQAref.")
        else:
            smoothing = getattr(args, "label_smoothing", 0.0)
            try:
                if smoothing and smoothing > 0.0:
                    model.celoss = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=smoothing)
                    print(f"Applied CrossEntropyLoss with label_smoothing={smoothing} and class weights.")
                else:
                    model.celoss = nn.CrossEntropyLoss(weight=weights_tensor)
                    print("Applied class weights to CrossEntropyLoss.")
            except TypeError:
                if smoothing and smoothing > 0.0:
                    print("Warning: label_smoothing not supported by current PyTorch; proceeding without smoothing.")
                model.celoss = nn.CrossEntropyLoss(weight=weights_tensor)
    except Exception as err:
        print(f"Warning: failed to compute/apply class weights for PQAref: {err}")

    train_node_classifier(
        G,
        labels_list,
        model,
        args,
        multimodal_loader,
        writer=writer,
        splits=splits,
    )


def mutag_task(args, writer=None):
    dataset_name = "MUTAG"
    processed_dir = Path("data") / dataset_name / "processed"
    graph_path = processed_dir / "mutag_graph.gpickle"
    labels_path = processed_dir / "mutag_labels.csv"
    splits_path = processed_dir / "split_masks.npz"
    fallback_pkl = Path("data") / f"{dataset_name}.pkl"

    if graph_path.exists() and labels_path.exists():
        G = nx.read_gpickle(graph_path)
        labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    elif fallback_pkl.exists():
        with open(fallback_pkl, "rb") as fh:
            G, labels_list, _ = pickle.load(fh)
            labels_list = list(map(int, labels_list))
        print("Warning: falling back to pickle load for MUTAG. Run scripts/prepare_mutag.py for up-to-date artefacts.")
    else:
        raise FileNotFoundError(
            "Processed MUTAG artefacts not found. Please run 'python scripts/prepare_mutag.py' first."
        )

    splits = None
    if splits_path.exists():
        npz = np.load(splits_path)
        splits = {key: np.asarray(npz[key]).astype(np.int64) for key in npz.files}
    else:
        print("Warning: split_masks.npz missing for MUTAG; falling back to random node splits during training.")

    labels_np = np.asarray(labels_list, dtype=int)
    if labels_np.ndim != 1:
        labels_np = labels_np.reshape(-1)
    num_classes = int(labels_np.max()) + 1 if labels_np.size > 0 else 0
    if num_classes <= 1:
        raise RuntimeError("MUTAG dataset requires at least two classes to train a classifier.")

    sample_node = next(iter(G.nodes()))
    feat_value = G.nodes[sample_node]["feat"]
    input_dim = len(feat_value) if isinstance(feat_value, (np.ndarray, list, tuple)) else int(feat_value)

    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=args.gpu)

    if args.multimodal:
        model = MultimodalGcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method,
            force_text_fallback=args.force_text_fallback,
            disable_text=args.disable_text,
            disable_image=args.disable_image,
        )
    else:
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

    device = torch.device(f"cuda:{args.cuda}" if args.gpu and torch.cuda.is_available() else "cpu")
    args.device = device
    model = model.to(device)

    try:
        total_samples = labels_np.shape[0]
        class_counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        eps = 1e-6
        weights = total_samples / (float(num_classes) * (class_counts + eps))
        max_w = getattr(args, "max_class_weight", 50.0)
        weights = np.minimum(weights, max_w)
        weights[class_counts == 0] = 0.0
        weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        ls = getattr(args, "label_smoothing", 0.0)
        try:
            model.celoss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=ls)
            if ls:
                print(f"Applied class weights with label_smoothing={ls} for MUTAG.")
        except TypeError:
            model.celoss = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Applied class weights for MUTAG (balanced heuristic).")
    except Exception as exc:
        print(f"Warning: unable to configure class weights for MUTAG ({exc}). Using default loss.")

    train_node_classifier(
        G,
        labels_list,
        model,
        args,
        multimodal_loader,
        writer=writer,
        splits=splits,
    )


def nlm_chem_task(args, writer=None):
    dataset_name = "nlm_chem"
    processed_dir = Path("data") / dataset_name / "processed"
    graph_path = processed_dir / "nlm_chem_graph.gpickle"
    labels_path = processed_dir / "nlm_chem_labels.csv"
    fallback_pkl = Path("data") / f"{dataset_name}.pkl"

    if graph_path.exists() and labels_path.exists():
        G = nx.read_gpickle(graph_path)
        labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    elif fallback_pkl.exists():
        with open(fallback_pkl, "rb") as fh:
            G, labels_list, _ = pickle.load(fh)
            labels_list = list(map(int, labels_list))
    else:
        raise FileNotFoundError(
            "Processed NLM-Chem artifacts not found. Please run 'python scripts/prepare_nlm_chem.py' first."
        )

    if len(G) == 0:
        raise RuntimeError("Loaded NLM-Chem graph is empty.")

    labels_np = np.array(labels_list, dtype=int)
    num_classes = int(labels_np.max()) + 1 if labels_np.size else 0
    if num_classes <= 1:
        raise RuntimeError(
            "NLM-Chem dataset appears to have <=1 class. Check preprocessing or bucket strategy."
        )

    sample_node = next(iter(G.nodes()))
    input_dim = len(G.nodes[sample_node]["feat"])

    import scipy.sparse as sp

    adj_sparse = nx.to_scipy_sparse_matrix(G, dtype=np.float32, format="coo")
    feat_array = np.array([G.nodes[i]["feat"] for i in G.nodes], dtype=np.float32)

    if args.multimodal:
        multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=args.gpu)
        model = MultimodalGcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            text_dim=getattr(args, "text_dim", 300),
            image_dim=getattr(args, "image_dim", 128),
            fusion_method=getattr(args, "fusion_method", "concat"),
            disable_text=args.disable_text,
            disable_image=args.disable_image,
        )
    else:
        multimodal_loader = None
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

    device = torch.device(f"cuda:{args.cuda}" if args.gpu and torch.cuda.is_available() else "cpu")
    args.device = device
    model = model.to(device)

    # Balanced class weights (optional)
    try:
        total_samples = labels_np.shape[0]
        counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        eps = 1e-6
        weights = total_samples / (num_classes * (counts + eps))
        max_w = getattr(args, "max_class_weight", 25.0)
        weights = np.minimum(weights, max_w)
        weights[counts == 0] = 0.0
        weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        ls = getattr(args, "label_smoothing", 0.0)
        try:
            model.celoss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=ls)
        except TypeError:
            model.celoss = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"Applied class weights for NLM-Chem (max={weight_tensor.max().item():.2f}).")
    except Exception as exc:
        print(f"Warning: unable to set class weights for NLM-Chem ({exc}). Using default loss.")

    def sparse_to_torch(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)).to(device)
        values = torch.from_numpy(sparse_mx.data).to(device)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    adj = sparse_to_torch(adj_sparse)
    x = torch.from_numpy(feat_array).to(device)

    G.graph["adj"] = adj
    G.graph["x"] = x

    train_node_classifier(G, labels_list, model, args, multimodal_loader, writer=writer)


def pkl_task(args, feat=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name=args.pkl_fname.split('.')[0],
            gpu=args.gpu
        )

    with open(os.path.join(args.datadir, args.pkl_fname), "rb") as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph["label"] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph["label"] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
        graphs, args, test_graphs=test_graphs
    )

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=args.num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            force_text_fallback=args.force_text_fallback,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method,
            disable_text=args.disable_text,
            disable_image=args.disable_image,
        )
    else:
        model = models.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn
        )

    train(
        train_dataset, model, args, multimodal_loader,
        val_dataset=val_dataset, test_dataset=test_dataset
    )
    evaluate(test_dataset, model, args, multimodal_loader, "Validation")


def enron_task_multigraph(args, idx=None, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="enron",
            gpu=args.gpu
        )

    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            net.add_nodes_from(range(max_enron_id))
            labels = [n[1].get("role", "None") for n in net.nodes(data=True)]
            labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            labels_list.append(labels_num)

        # 构建模型
        if args.multimodal:
            model = MultimodalGcnEncoderNode(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.output_dim,
                label_dim=len(labels_dict),
                num_layers=args.num_gc_layers,
                bn=args.bn,
                args=args,
                force_text_fallback=args.force_text_fallback,
                text_dim=args.text_dim,
                image_dim=args.image_dim,
                fusion_method=args.fusion_method,
                disable_text=args.disable_text,
                disable_image=args.disable_image,
            )
        else:
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, len(labels_dict),
                args.num_gc_layers, bn=args.bn, args=args
            )

        if hasattr(args, 'device'):
            model = model.to(args.device)
        print(labels_num)
        train_node_classifier_multigraph(
            G_list, labels_list, model, args, multimodal_loader, writer=writer
        )
    else:
        print("Running Enron full task")


def enron_task(args, idx=None, writer=None):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name="enron",
            gpu=args.gpu
        )

    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)

        G = nx.disjoint_union_all(G_list)
        
        # 构建模型
        if args.multimodal:
            model = MultimodalGcnEncoderNode(
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.output_dim,
                label_dim=len(labels_dict),
                num_layers=args.num_gc_layers,
                bn=args.bn,
                args=args,
                force_text_fallback=args.force_text_fallback,
                text_dim=args.text_dim,
                image_dim=args.image_dim,
                fusion_method=args.fusion_method,
                disable_text=args.disable_text,
                disable_image=args.disable_image,
            )
        else:
            model = models.GcnEncoderNode(
                args.input_dim, args.hidden_dim, args.output_dim, len(labels_dict),
                args.num_gc_layers, bn=args.bn, args=args
            )

        labels = [n[1].get("role", "None") for n in G.nodes(data=True)]
        labels_num = [labels_dict[l] for l in labels]
        for i in range(5):
            print("Label ", i, ": ", labels_num.count(i))

        print("Total num nodes: ", len(labels_num))
        print(labels_num)

        if hasattr(args, 'device'):
            model = model.to(args.device)
        train_node_classifier(G, labels_num, model, args, multimodal_loader, writer=writer)
    else:
        print("Running Enron full task")


def benchmark_task(args, writer=None, feat="node-label"):
    # 初始化多模态加载器
    multimodal_loader = None
    if args.multimodal:
        multimodal_loader = MultimodalDataset(
            dataset_name=args.bmname,
            gpu=args.gpu
        )

    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )
    print(max([G.graph["label"] for G in graphs]))

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
        graphs, args, max_nodes=args.max_nodes
    )

    # 构建模型
    if args.multimodal:
        model = MultimodalGcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=args.num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
            force_text_fallback=args.force_text_fallback,
            text_dim=args.text_dim,
            image_dim=args.image_dim,
            fusion_method=args.fusion_method,
            assign_ratio=args.assign_ratio if args.method == "soft-assign" else None,
            num_pooling=args.num_pool if args.method == "soft-assign" else None,
            dropout=args.dropout,
            linkpred=args.linkpred,
            assign_input_dim=assign_input_dim if args.method == "soft-assign" else None,
            disable_text=args.disable_text,
            disable_image=args.disable_image,
        )
    elif args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes, input_dim, args.hidden_dim, args.output_dim,
            args.num_classes, args.num_gc_layers, args.hidden_dim,
            assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred,
            args=args, assign_input_dim=assign_input_dim
        )
    else:
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args
        )
    # move model to device if provided
    if hasattr(args, 'device'):
        model = model.to(args.device)

    train(
        train_dataset, model, args, multimodal_loader,
        val_dataset=val_dataset, test_dataset=test_dataset, writer=writer
    )
    evaluate(test_dataset, model, args, multimodal_loader, "Validation")


def benchmark_task_val(args, writer=None, feat="node-label"):
    all_vals = []
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # 10折交叉验证
    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
            graphs, args, i, max_nodes=args.max_nodes
        )
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args
        )
        if hasattr(args, 'device'):
            model = model.to(args.device)

        _, val_accs = train(
            train_dataset, model, args,
            val_dataset=val_dataset, test_dataset=None, writer=writer
        )
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphPool arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    # 多模态相关参数
    multimodal_parser = parser.add_argument_group()
    multimodal_parser.add_argument(
        "--multimodal",
        dest="multimodal",
        action="store_const",
        const=True,
        default=False,
        help="Whether to use multimodal features"
    )
    multimodal_parser.add_argument(
        "--text-dim",
        dest="text_dim",
        type=int,
        default=300,
        help="Dimension of text features"
    )
    multimodal_parser.add_argument(
        "--image-dim",
        dest="image_dim",
        type=int,
        default=2048,
        help="Dimension of image features"
    )
    multimodal_parser.add_argument(
        "--fusion-method",
        dest="fusion_method",
        type=str,
        default="concat",
        choices=["concat", "add", "attention", "bilinear"],
        help="Method for fusing multimodal features"
    )
    multimodal_parser.add_argument(
        "--force-text-fallback",
        dest="force_text_fallback",
        action="store_true",
        default=False,
        help="Force using tokenizer/hashing text embedding fallback instead of loading transformers models",
    )
    multimodal_parser.add_argument(
        "--disable-text",
        dest="disable_text",
        action="store_const",
        const=True,
        default=False,
        help="Disable text modality in multimodal fusion"
    )
    multimodal_parser.add_argument(
        "--disable-image",
        dest="disable_image",
        action="store_const",
        const=True,
        default=False,
        help="Disable image modality in multimodal fusion"
    )

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument(
        "--assign-ratio",
        dest="assign_ratio",
        type=float,
        help="ratio of number of nodes in consecutive layers",
    )
    softpool_parser.add_argument(
        "--num-pool", dest="num_pool", type=int, help="number of pooling layers"
    )
    parser.add_argument(
        "--linkpred",
        dest="linkpred",
        action="store_const",
        const=True,
        default=False,
        help="Whether link prediction side objective is used",
    )

    parser_utils.parse_optimizer(parser)

    parser.add_argument(
        "--datadir", dest="datadir", help="Directory where benchmark is located"
    )
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
        "--max-nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="Ratio of number of graphs training set to all graphs.",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument(
        "--input-dim", dest="input_dim", type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-classes", dest="num_classes", type=int, help="Number of label classes"
    )
    parser.add_argument(
        "--merge-rare-classes",
        dest="merge_rare_classes",
        type=int,
        default=0,
        help="Merge classes with fewer than this many examples into an 'Other' class (0=disabled)",
    )
    parser.add_argument(
        "--use-focal",
               dest="use_focal",
        action="store_const",
        const=True,
        default=False,
        help="Use Focal Loss instead of CrossEntropyLoss",
    )
    parser.add_argument(
        "--weight-scheme",
        dest="weight_scheme",
        type=str,
        default="balanced",
        choices=["balanced", "inv_sqrt", "log"],
        help="Scheme to compute per-class weights: balanced (total/(K*count)), inv_sqrt (1/sqrt(count)), log (1/log(1+count))",
    )
    parser.add_argument(
        "--max-class-weight",
        dest="max_class_weight",
        type=float,
        default=50.0,
        help="Clip per-class weights to this maximum to avoid instability",
    )
    parser.add_argument(
        "--focal-gamma",
        dest="focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss when --use-focal is enabled",
    )
    parser.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value for CrossEntropyLoss (0 disables)",
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
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument(
        "--method", dest="method", help="Method. Possible values: base, "
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=20,
        num_epochs=1000,
        train_ratio=0.8,
        test_ratio=0.1,
        num_workers=1,
        input_dim=10,
        hidden_dim=20,
        output_dim=20,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        name_suffix="",
        assign_ratio=0.1,
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        # set device attribute for callers
        if torch.cuda.is_available():
            prog_args.device = torch.device("cuda")
            # enable cuDNN autotuner for potentially better performance
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            except Exception:
                pass
            print("CUDA", prog_args.cuda)
        else:
            # CUDA not available despite user requesting GPU
            prog_args.device = torch.device("cpu")
            print("CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
    else:
        prog_args.device = torch.device("cpu")
        print("Using CPU")

    # 使用多模态时打印配置
    if prog_args.multimodal:
        print(f"Using multimodal features - text_dim: {prog_args.text_dim}, image_dim: {prog_args.image_dim}, fusion: {prog_args.fusion_method}")

    # 任务路由
    if prog_args.bmname is not None:
        benchmark_task(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        ds_key = str(prog_args.dataset).lower()
        if ds_key == "syn1":
            syn_task1(prog_args, writer=writer)
        elif ds_key == "pqaref":
            pqaref_task(prog_args, writer=writer)
        elif ds_key == "mutag":
            mutag_task(prog_args, writer=writer)
        elif ds_key == "syn2":
            syn_task2(prog_args, writer=writer)
        elif ds_key == "syn3":
            syn_task3(prog_args, writer=writer)
        elif ds_key == "syn4":
            syn_task4(prog_args, writer=writer)
        elif ds_key == "syn5":
            syn_task5(prog_args, writer=writer)
        elif ds_key == "enron":
            enron_task(prog_args, writer=writer)
        elif ds_key == "ppi_essential":
            ppi_essential_task(prog_args, writer=writer)
        elif ds_key == "nlm_chem":
            nlm_chem_task(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()
