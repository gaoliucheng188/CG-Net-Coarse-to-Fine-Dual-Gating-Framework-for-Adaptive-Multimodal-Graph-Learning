#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline对比实验训练脚本

在服务器上运行:
    python baselines/run_comparison.py --dataset MUTAG --epochs 100

输出结果包含:
    - 训练/测试准确率
    - 训练时间
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_multimodal import MultimodalGcnEncoderGraph


def load_dataset(dataset_name):
    """加载数据集"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    pkl_path = os.path.join(data_dir, f'{dataset_name}.pkl')

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    G = data[0]
    labels = np.array(data[1])

    # 构建邻接矩阵
    n = G.number_of_nodes()
    adj = np.zeros((n, n))
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1

    # 节点特征
    node_list = list(G.nodes())
    feat_dim = G.nodes[node_list[0]]['feat'].shape[0]
    features = np.zeros((n, feat_dim))
    for i, node in enumerate(node_list):
        features[i] = G.nodes[node]['feat']

    return adj, features, labels


def load_text(dataset_name):
    """加载文本"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'texts')
    text_path = os.path.join(data_dir, dataset_name, 'graph_0.txt')

    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


class DummyArgs:
    """模拟args对象"""
    def __init__(self):
        self.gpu = True
        self.method = 'base'
        self.bias = True


def train_model(model, x, adj, labels, text, train_idx, test_idx, epochs, lr):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
    adj_t = torch.FloatTensor(adj).unsqueeze(0).to(device)
    labels_t = torch.LongTensor(labels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_t, adj_t, text=text)
        loss = criterion(outputs, labels_t)

        loss.backward()
        optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            train_acc = accuracy_score(labels[train_idx], preds[train_idx].cpu())
            test_acc = accuracy_score(labels[test_idx], preds[test_idx].cpu())

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

    elapsed = time.time() - start_time
    return best_test_acc, elapsed


def run_experiment(dataset_name, method, epochs=100, lr=0.001, hidden_dim=64, embed_dim=64):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Method: {method}")
    print(f"{'='*50}")

    # 加载数据
    adj, features, labels = load_dataset(dataset_name)
    text = load_text(dataset_name)

    n = len(labels)
    num_classes = len(np.unique(labels))
    input_dim = features.shape[1]

    print(f"  Nodes: {n}, Classes: {num_classes}, Input dim: {input_dim}")

    # 划分数据集
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    args = DummyArgs()

    # 创建模型
    if method == 'Always_Full':
        model = MultimodalGcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embed_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
            fusion_method='concat',
            disable_text=False,
            args=args,
        )
    elif method == 'Drop_Text':
        model = MultimodalGcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embed_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
            fusion_method='concat',
            disable_text=True,  # 禁用文本
            args=args,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # 训练
    acc, elapsed = train_model(model, features, adj, labels, text, train_idx, test_idx, epochs, lr)

    print(f"  Best Test Acc: {acc:.4f} | Time: {elapsed:.1f}s")

    return {'method': method, 'accuracy': acc, 'time': elapsed}


def main():
    parser = argparse.ArgumentParser(description='Baseline对比实验')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                       choices=['MUTAG', 'nlm_chem', 'pqaref'],
                       help='数据集名称')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'Always_Full', 'Drop_Text'],
                       help='对比方法: all=两个都跑')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--embed', type=int, default=64, help='嵌入维度')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CG-Net Baseline Comparison Experiments")
    print("="*60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Hidden: {args.hidden}, Embed: {args.embed}")
    print("="*60)

    results = []

    if args.method == 'all':
        methods = ['Always_Full', 'Drop_Text']
    else:
        methods = [args.method]

    for method in methods:
        result = run_experiment(
            args.dataset, method,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden,
            embed_dim=args.embed,
        )
        results.append(result)

    # 打印汇总
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"{'Method':<15} | {'Accuracy':>10} | {'Time(s)':>10}")
    print("-"*40)
    for r in results:
        print(f"{r['method']:<15} | {r['accuracy']:>10.4f} | {r['time']:>10.1f}")
    print("="*60)

    # 保存结果到文件
    result_file = f"baseline_results_{args.dataset}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}\n\n")
        for r in results:
            f.write(f"{r['method']}: {r['accuracy']:.4f}\n")
    print(f"\nResults saved to: {result_file}")


if __name__ == '__main__':
    main()

