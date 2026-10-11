# CG-Net 重投 ESWA 整改方案
> 生成日期：2026-10-05
> 依据：AAAI-27 Phase-1 三位评审（ctjR29 / fndf / AI Reviewer）意见 + 当前仓库代码与 paper.tex。
> 目标期刊：*Expert Systems With Applications* (Elsevier)。

---

## 0. 结论（先读这个）

**值得重投 ESWA，但"明天投"不现实；务实可行的时间是 1–2 周。** 理由：

1. **三位评审一致判死的那条"测试标签泄漏"，是写作/符号问题，不是方法缺陷。** 代码里 Soft Masker 掩码优化目标是**保留模型自己的预测类**（见下文证据），测试时**没有用真实标签**。这可以直接靠改写救回来，且是投入产出比最高的一处。ESWA 大概率不会再因这条被拒。
2. **但还有 3 类真实短板，写文字救不回来**：MUTAG 干净文本下 CG-Net 输给 GIN、Hard Router 不贡献精度、NLM-Chem 只有 30 个测试节点。对这几条要有诚实策略（见 §3）。
3. ESWA 是 Elsevier，格式与 AAAI 完全不同（双栏模板、页数约束），且当前仓库**缺全部 Figures**，格式改造本身就要时间。

## 0.5 🔴 数据完整性风险（重投前必须优先解决）

**2026-10-05 核查结论：论文的最终实验数字在本机仓库无法复现，且本地存在多组互相矛盾的运行结果。**

| 来源 | MUTAG @ τ=0.7 CG-Net | bypass | 是否匹配论文 |
|---|---|---|---|
| 论文 Table 1 | **0.8426** | 189/502 | — |
| 本机 `eval_mutag_tau0.7.out` | 0.7390 | 50/502 | ✗ |
| 本机 `multiseed_detailed.json`（seed42） | 0.8347 | 189/502 | bypass ✓ / Acc ✗ |
| 本机 `gin_5s_42.log`（GIN 基线） | 0.8187 | — | ✗（论文写 0.779） |

- 论文的 0.8426、0.779、0.935 **在本机任何文件里都搜不到**；权威运行在 Linux 服务器（`/home/liuchenggao/.../master1`）。
- **已确认决策（用户 2026-10-05, 2026-10-06 修正）**：
  - **Q1 τ=0.7 是"验证集扫描选定"** → A5 可写"按验证集扫描，精度不掉前提下最大化绕行率选 τ，未用测试集调参"。
  - **Q2 采用"本机重跑对齐"（用户已拍板：本地数字 + 诚实重写）**。规范源：**本机 `evaluate_explainer_gating.py` 复现数字**（非当年 multiseed JSON）。见下方"规范数字"子表。
  - **Q3 GLANCE 采用"（a）补 related work +（b）加简化对照实验"**。

### 规范数字（本机复现，τ=0.7, full_modal ckpt, PYTHONHASHSEED=42, 20 explainer epochs）

| 数据集 | Always Full | Drop Text | Hard Only | Soft Only | CG-Net(OURS) | bypass |
|---|---|---|---|---|---|---|
| MUTAG | **0.6494** | 0.7371 | 0.6494 | **0.7629** | **0.7629** | 189/502 (37.6%) |
| NLM-Chem | 0.9667 | 1.0 | 0.9667 | 0.9667 | 0.9667 | 27/30 (90%) |
| PQA-Ref | 0.9349 | 0.9349 | 0.9349 | 0.9349 | 0.9349 | 236/307 (76.9%) |

**逐条核验（2026-10-06）：**
- **NLM、PQA 本地与 multiseed JSON 完全吻合**（确定性数字：full/notext/hard/bypass 逐一一致；PQA 全 = 0.9349、bypass 236 一致；NLM full 0.9667/notext 1.0/bypass 27 一致）→ **文本与纯图分支两端一致，本地管线可信**。
- **MUTAG 分歧仅在"图+文本+图像"融合路径**：本地 full=0.6494 vs multiseed 0.7789；CG-Net 0.7629 vs 0.8347。但 DropText(0.7371)、bypass(189)、文本编码(NLM 证)、图像特征(DropText 证) 全部两端一致 → 分歧源头是 **6/2 evaluate 脚本在 MUTAG 图像融合路径的行为差异（非 git，无法还原）**，**非本地可修 bug**（已验证 ImageEncoder/ResNet 正常、图像特征与 multiseed 一致）。
- **诚实定位**：MUTAG 上 CG-Net 0.7629 主数字为**本机确定可复现**；multiseed 的 0.8347 本机无法复现。NLM/PQA 数字可与服务器交叉印证。

### 本地复现环境修复（已就位，勿回退）
- `utils/__init__.py`：纯 `pickle` 实现 `nx.read_gpickle`/`write_gpickle`（networkx 3.2.1 移除 gpickle）。
- `multimodal/text_encoder.py`：先试普通 `from_pretrained(local_files_only=True)`，再兜底 `low_cpu_mem_usage`（避免 `accelerate` 依赖与联网）。
- `bert-base-uncased/pytorch_model.safetensors`：由 `.bin` 转换，绕过 torch 2.0.1 对 `.bin` 的加载安全限制（保留 `convert_bert_bin_to_safetensors.py`）。
- 8GB 显存限制：一次只跑一个数据集（PQA 需 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`）。

**⚠️ 本机运行注意**：代码为 networkx 2.x 编写；已在 `utils/__init__.py` 加兼容垫片恢复 `nx.read_gpickle`（不动环境）。运行环境 `conda activate pytorch`（torch 2.0.1+cu118，RTX 4060 Laptop 8GB，比服务器 24GB 小，重训大模型可能 OOM，优先复用现有 `ckpt/*_full_modal.pth.tar`）。

---

## 1. 最重要的好消息：泄漏问题可以救（先做）

### 评审怎么说的
- fndf（clear rejection）核心论据："mask 在 test-time 优化，目标却含 ground-truth 类（Eq.1 = `-log ŷ_y`）。若只在训练用则与'测试时优化'矛盾。"
- AI Reviewer 扩展：Eq.2 用 `log(M)`，而 `sigma(0)=0.5` 说明 `M` 像 logit，logits 的 log 未定义——目标自相矛盾。

### 代码实际怎么做的（证据）
`explainer/explain_multimodal.py:165-196` `loss()`：
- 注释原文：*"optimize masks to preserve the model's OWN predicted class, never the ground-truth label. Using GT here would leak test labels into Soft Masker fusion weights."*
- 目标类来自 `pred_label = self._gather_node_predictions(neighbors)`，其数据源是 `pred_fake_np = prob_full_all`（**模型自己全模态 softmax 输出**），不是 `test_label_np`。
- `evaluate_explainer_gating.py:378` 融合 `soft=(1-s_T)·P_G + s_T·P_GT`，`s_T` 由无标签优化的标量掩码得出。

⇒ **结论：测试时整条链路没有用真实标签。** 论文 `\eqref{eq:explainer_loss}` 里把预测损失写成 `-\log \hat p_{gt}`（"ground truth class"）是错的，致命。

### 一句话改法
把 Eq(2) 的 `\hat p_{gt}`（gt=真值）改成 **`\hat p_{\hat y}`，其中 `\hat y = argmax_y P_{GT}(y|v)`（模型自身预测类）**，并新增一句明确声明：

> *"Soft Masker is a label-free test-time procedure: the masks are optimized to preserve the model's own predicted class, so no ground-truth labels are used during inference, and all reported test accuracy is computed without label-adapted masks."*

同时修正 `\mathcal L_{size}=\sum|M|` 与 `\mathcal L_{ent}` 的记号不一贯：明确 `M = \sigma(m)`（sigmoid(原始 logit)），`\mathcal L_{size}` 对 `M` 求和、`\mathcal L_{ent}` 对 `M` 算熵，**避免出现"对 logits 取 log"的数学矛盾**。AI Reviewer 那条"logits vs probabilities"随之解决。

> ⚠️ 附加：**诚实性提示**——掩码目标是"保留模型自身预测"，因此它无法真的"修正模型的全模态错误预测"（文中 Case 326 "rectifies the wrong prediction" 的说法有一定夸大，见 §3.4）。建议把案例表述改成"在 graph 与 full 预测分歧时，通过 s_T 加权偏向更可靠的模态"。

---

## 2. 审稿意见逐条整改表

### A. 纯写作即可救(今晚可做)

| # | 评审·条目 | 现状(paper.tex) | 改法 | 状态 |
|---|---|---|---|---|
| A1 | fndf/AI：测试标签泄漏 | Eq(6) `\mathcal L_pred=-log \hat p_{gt}`；称"test-time optimization" | 见 §1，换成 `\hat y` + 标签无关声明 | 可救 |
| A2 | AI：Eq(1)"logit margin"与公式不符 | 正文说 "adopt logit margin"，但 Eq(1) 用 softmax 概率 `p_top1-p_top2` | 改成 "probability margin" | 可救 |
| A3 | fndf/AI：mask 是 logits 还是概率不自洽 | `sigma(0)=0.5` 但 `L_ent` 对 `M` 取 log | 声明 `M=sigma(m)`，损失统一对 `M` | 可救 |
| A4 | ctjR29：Eq(4)(5) 只写 `M_T` 贡献，`M_G` 去向未明 | 文中 M_G 只定义、不用在最终预测 | **诚实改写**：明确 `M_G` 用于可解释性/Fidelity 分析，不改变分类预测；只 `s_T` 影响融合(见 §3.3) | 可救(需诚实) |
| A5 | ctjR29/fndf：τ=0.7 选择标准未说明 | 只说 "according to threshold sweeping results" | 补充：在**验证集**上按"阈値扫 τ∈{0.1..0.9}，取使 AUC 边界的 Acc 不降且 bypass 尽量高"的准则选择，说明未用测试集调参 | 可救 |
| A6 | fndf：BERT 预训练矛盾 | 正文 "BERT-based sentence transformers" 又声明 "trained from scratch without external pretrained weights" | 明确文本编码器初始化与是否冻结(见 §3.5) | 可救(需核代码) |
| A7 | AI：NLM-Chem 表间不一致 | Table 1 Drop Text=1.0000，Table sota Drop Text=0.935；MUTAG 表间 0.7371 vs 0.697 | **统一数字**，标注"单 seed vs 5-seed 均值"口径 | 可救(需核数据) |
| A8 | 三评审：Figure 字号过小 | 图不在仓库 | 重绘/放大坐标轴与图例 | 需重绘 |
| A9 | AI：图3 画了平滑曲线+interfering regime，与 "未观测到 interference" 矛盾 | Table tau_sweep 只有 5 个点；正文说 interfering 未观测 | 图加 "schematic/l概念示意" 标注，或删 smooth curve | 需重绘 |

### B. 需要补实验才能救（可跑通，见 §4）

| # | 评审·条目 | 做法 | 现有脚本/数据 |
|---|---|---|---|
| B1 | fndf/AI：绕行率≠端到端效率；未含 Soft Masker 20 步优化成本 | 用 `run_latency_full_vs_vara.py` 重跑，**加入 Soft Masker 两步优化**，报端到端单样本延迟/吞吐/FLOPs | ∎ `run_latency_full_vs_vara.py`、`latency_full_vs_vara.out`(旧) |
| B2 | fndf：M_G 为稠密 |V|×|V| 可扩展性(3149 节点)→~10M mask | 明确 M_G 为**局部子图/稀疏边掩码**实现，并补内存/优化开销报告 | 需核实解释器实现 |
| B3 | fndf/fndf/ctjR29：缺与选择性 LLM 调用(GLANCE)对比 | related work 加 GLANCE + GraphBridge；理想补一组对照(条件路由) | 需实现/简化 GLANCE | 大工程，见 §4 |
| B4 | 三评审：需更稳/更显著的数据支撑 | MUTAG/NLM 改成 5-seed 均值±std 统一口径；补配对 t 检验(正文已有声明，需落地成表) | `run_multiseed_test.py`、`run_multiseed.sh` | 需跑 |

### C. 真实短板（无法靠文字救，需策略，见 §3）

| # | 短板 | 评审源头 |
|---|---|---|
| C1 | 干净 MUTAG 下 CG-Net 0.757 < GIN 0.779 | ctjR29 / fndf / AI |
| C2 | Hard Router 本身不贡献精度(MUTAG 0.7749 < Always Full 0.7789)；CG-Net=Soft Only | fndf(AI) |
| C3 | NLM-Chem 仅 30 测试节点；PQARef 全打平 | fndf / ctjR29 |
| C4 | Soft Masker 学的 s_T 作为"可解释性"意义有限 | fndf / AI |

---

## 3. 真实短板与诚实策略

这几条决定论文天花板，改写无效，必须用**诚实定位 + 框定贡献**来化解：

- **C1（MUTAG 输给 GIN）**：不要在正文争"我更强"。改为明确框架定位是**"精度–效率可调节的调度器"而非纯精度 SOTA**。论点：
  1) 在 NLM-Chem 上 CG-Net 显著优于 GIN/TextGCN/GLEM(1.0000 vs 0.987/0.960/0.956)；
  2) 在 MUTAG 干净文本下，CG-Net 比 Always Full +0.020，说明 Soft Masker 确从文本中提取了残差信号；
  3) 主卖点是用 37.6%–90% 的文本绕行换到**近乎无损**的精度。这样把评价指标从"最高 ACC"挪到"ACC×cost 联合收益"。
- **C2（Hard Router 不增精度）**：诚实承认 Hard Router 是**效率组件而非精度组件**，并把这变成卖点：它用 0 参数、0 训练、0 推理开销实现了每样本计算调度——这正是相对 RL/dynamic gating 的差异化贡献。Soft Masker 才是精度组件。二者解耦，JP：对照 Variant A/B 表。
- **C3（数据规模）**：NLM-Chem 30 节点是小数据集，诚实放入 limitation，并补一句"已在更大图(PQARef 3149 节点)上验证可扩展性(见 B2)"。
- **C4（s_T 可解释性）**：不再声称 s_T 衡量"语义可靠度"，改称"残余文本可信度/加权系数"，弱化语义解释，强化它是无标签学到的融合权重。

---

## 4. 需要补/修正的实验清单（priority 降序）

> 运行环境：`conda activate pytorch`（Windows 本机；GPU 重训建议仍在你服务器跑，路径含 `/home/liuchenggao/...` 的是旧路径，需改）。

1. **[P0] 端到端时序含 Soft Masker**：改 `run_latency_full_vs_vara.py`，对 hard 样本计入 20 步反向传播，输出单样本 end-to-end 延迟与吞吐（回应 B1）。**这是判定"90% 减算"到底打几折的关键。**
2. **[P0] 数字一致性核查**：导出表格所有数字，统一 MUTAG/NLM 的 Drop Text 口径（回应 A7）。
3. **[P1] 5-seed 统一口径 + 配对 t 检验落表**（回应 B4）。
4. **[P1] M_G 可扩展性**：核实解释器 M_G 是否为稀疏/局部掩码，补内存与优化成本报告（回应 B2）。
5. **[P2] GLANCE 简化对照**：实现一个"按置信度决定是否调用 text 编码"的对照路由（不实现全套 GLANCE），进 comparison 表或至少 related work 讨论（回应 B3）。这一步若时间紧可不做实验、仅补 related work + limitation 讨论。

---

## 5. ESWA 格式改造清单（AAAI → Elsevier）

> **2026-10-06 已产出 `eswa_main.tex`**（复制自 paper.tex 后转换，paper.tex 原件保留；无 git 故未原地改）。本机无 LaTeX，需服务器/Overleaf 编译验证。

- [x] 替换 `documentclass`：`article+aaai2027` → `\documentclass[preprint,12pt,authoryear]{elsarticle}`（已做；review 行距用 `[preprint,review,12pt]`）。
- [x] 版式：去掉 times/helvet/courier/aaai2027/algorithm/newfloat/listing（正文未用）；作者/单位改 `\author`+`\affiliation`+`\ead`+`\corref`（先占位 Anonymous，投稿前补真名）。
- [~] 参考文献：已加 `\bibliographystyle{elsarticle-harv}`（按用户提供的 harvard 模板）。**注意 ESWA 官方常要求 numeric `model1-num-names`**，投稿前确认；30 个引用键已核验全部在 refs.bib，无 undefined。
- [x] 结构合规：Intro/Related work/Methodology/Experiments/Conclusion 保留；已加 **highlights** + **keyword**（Elsevier 要求）。
- [ ] 页数：ESWA 双栏 ~15–20 页；preprint 单栏偏长，final 用 `[final,3p,twocolumn]` 后视页数压缩。
- [x] 摘要：已加 Keywords + Highlights。
- [ ] 图表重绘（A8/A9）并写入常规分辨率要求（用户负责）。
- [ ] **待用户补**：真实作者名/单位、通讯作者邮箱；确认引用风格（harvard vs model1-num-names）。

---

## 6. 材料缺口清单（你需提供，我改不了）

- [ ] **aaai2027.bib**（或全部参考文献条目）——正文引用了大量 2025/2026 名为 `citep{...}` 的条目，全都没定义。
- [ ] **Figures/** 目录：`motivation.png`、`diagrame.png`、`fig_paradigms.png`、`case_hard_105/120/2842.png`、`case_soft_326.png`。
- [ ] **训练好的 checkpoint** 位置（用于重跑时序/多 seed；`ckpt/*.pth.tar` 是否在本机）。
- [ ] 文本编码器初始化细节（冻结？预训练？），用于 A6。

---

### §6.x 写作类已完成（2026-10-06，paper.tex）

- [x] **A5 τ 选择**（line 154）：改为"验证集扫描 τ∈{0.1..0.9}，取验证 Acc 不降前提下最大化 text-bypass 的 τ，测试集不参与选阈"。
- [x] **case study 诚实化**（line 365）：删"rectifies original wrong prediction"，改"经 $s_T$ 抑制噪声文本、重加权图/文融合以偏向更可靠模态"。
- [x] **GLANCE/GraphBridge 对照节** `sec:selective_comp`（Conclusion 前）：用"仅路由对照（Variant A）"作选择性语言处理的诚实代理，对比路由 vs 路由+精化，无新 GPU 实验；line 117 转引同步软化为"routing-only selective-processing control"。
- [x] **环境行决策**：用户拍板统一写 RTX 4090/Ubuntu/torch 2.2（line 264 不改）；MUTAG 本机复现的透明声明放表格/C1 处。
- [ ] 待规范批次（b9uoz80hm）完成后：重建 Tab1 消融、Tab2 τ 扫描、multiseed 表、配对 t 检验，及 C1 诚实论述（用本地 MUTAG 0.6494/0.7371/0.7629）。

> **2026-10-06 方法级转移（重要，用户已拍板）**：可复现数字揭示旧 Soft Masker 掩码目标锚定「全融合自身预测」，导致 s_T→0.94 放大有害文本，MUTAG CG-Net 钉死在全融合地板 0.6494。已重设计为**模态安全融合**：掩码目标锚定**图分支(Drop-Text)预测类 ŷ_G**，文本与图冲突时压 s_T→0（回落到图分支≈Drop Text 0.7371）、文本强化图答案时才让 s_T 上升。改动：`evaluate_explainer_gating.py` 传 `pred_graph=prob_notext_all.unsqueeze(0).numpy()`；`explainer/explain.py` `__init__` 加 `pred_graph`、`_gather_node_predictions/_slice_prediction_scores` 加 `source` 参数；`explainer/explain_multimodal.py` `__init__` 接 `pred_graph`、`explain()` 用 `source="graph"` 作目标。同时给 `evaluate_explainer_gating.py` 加了 `--seed`(回退读 PYTHONHASHSEED)→torch/numpy/cudnn 全种子化(修掉旧哈希噪声)。诚实预期：新 CG-Net≤max(Full,Drop)，MUTAG≈0.737、NLM≈1.0，**不声称超过最优单模态**，主卖点=模态安全+自适应效率。

---

## 7. 建议时间表（务实完整版）

- **今晚**：§1 泄漏改写 + §2A 全部写作类修改（我可以直接改 paper.tex）→ 产出"逐条回应评审"初稿。
- **第1–3天**：P0/P1 实验（时序含 Soft Masker、数字核查、5-seed、M_G 可扩展性）跑通并回填表/图。
- **第4–7天**：补 GLANCE 讨论、ESWA 格式改造、写逐条 Response。
- **第8–14天**：多轮润色 + 补交缺的 .bib/Figures + 生成 ESWA 双栏版。
- **第14天前后**：投稿（Elsevier Editorial Manager 提交，含 ESWA 共同作者/作者贡献/author statement 等必填项）。