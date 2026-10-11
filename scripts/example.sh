# 训练多模态预测模型
python -m train --dataset=syn1 --multimodal --fusion-method=concat
python -m train --dataset=syn2 --multimodal --fusion-method=attention
python -m train --dataset=syn4 --multimodal --fusion-method=bilinear
python -m train --dataset=syn5 --multimodal


# 使用多模态解释器
python -m explainer_main --explain-node=300 --dataset=syn1 --multimodal  # 节点解释（基础图大小为300，可将explain-node改为301、302等）
python -m explainer_main --graph-idx=0 --dataset=syn2 --multimodal --graph-mode  # 图解释（基础图大小为350，前2个特征维度为有用特征，其余为非信息性特征）
python -m explainer_main --multinode-class=1 --dataset=syn4 --multimodal  # 多节点类解释（基础树大小为512，可将相关节点参数改为512等）
python -m explainer_main --multigraph-class=0 --dataset=syn5 --multimodal --graph-mode  # 多图类解释（基础树大小为512）