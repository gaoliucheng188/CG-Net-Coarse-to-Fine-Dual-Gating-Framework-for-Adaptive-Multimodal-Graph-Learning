# save_syn1.py
import pickle
from utils import featgen
from utils import gengraph
import numpy as np

# 生成 syn1 数据
G, labels, name = gengraph.gen_syn1(
    feature_generator=featgen.ConstFeatureGen(np.ones(10, dtype=float))  # 10可改为你的input_dim
)

# 保存为 pkl 文件
with open('data/syn1.pkl', 'wb') as f:
    pickle.dump((G, labels, name), f)
print("syn1.pkl 已生成")