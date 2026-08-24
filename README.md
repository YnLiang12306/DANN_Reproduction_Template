# 基于域对抗训练（DANN）的跨被试运动想象脑电分类 —— 复现作业

## 1. 作业背景

本次作业是脑电信号深度学习系列的进阶环节，主题是**无监督域适配（Unsupervised Domain Adaptation, UDA）**。在前面的复现中，你已经逐步掌握了**EEGNet、EEGConformer、SimCLR**

而本次要复现的是 **DANN（Domain-Adversarial Training of Neural Networks）** 尝试将其他领域的框架迁移到脑电中。

### 为什么跨被试是一个「域适配」问题？

脑电信号存在很强的**被试间差异（inter-subject variability）**：不同被试的头皮形状、电极位置、肌肉伪迹、精神状态、以及 ERD/ERS 的空间分布都不一样。这意味着：

- 在一个被试（源域）上训练的分类器，直接拿到另一个被试（目标域）上，精度往往会大幅下降；
- 目标被试往往没有（或只有极少的）标注数据，无法直接训练。

这正是域偏移问题：源域和目标域的数据分布不同（$P_s(x) \neq P_t(x)$），但任务相同（都是 4 类运动想象分类）。DANN 的目标是：**只用源域标签 + 无标签目标域数据，学到一个「域不变」（domain-invariant）的特征表示**，使得在源域上训练的分类器也能很好地泛化到目标域。

### DANN 的核心思想：对抗

DANN 借鉴了 GAN 的对抗思想，但对抗的不是「真假」，而是「域」：

- **特征提取器 $G_f$** 希望学到的特征既有利于分类，又「分不清来自哪个被试」；
- **域判别器 $G_d$** 则努力分辨特征到底来自源域还是目标域；
- 通过一个**梯度反转层（Gradient Reversal Layer, GRL）**，让 $G_f$ 与 $G_d$ 形成一个 min-max 对抗博弈。

当这个博弈达到平衡时 **$G_f$** 学到的特征就是**域不变的**，此时源域的分类器可以直接迁移到目标域。

本次评价重点是：**对域偏移、对抗训练、梯度反转层（GRL）机制的理解**，以及**能否用 DANN 把之前复现过的 EEGNet / EEGConformer 改造成一个跨被试可迁移的模型**。

## 2. 任务目标

复现 DANN，实现运动想象脑电的**跨被试（无监督域适配）4 分类**。

任务形式：

```text
输入：
  源域（带标签）：8 名被试的运动想象 EEG（22 通道 × 时间点）
  目标域（无标签）：1 名被试的运动想象 EEG（标签仅用于最终评估，训练时不可见）
输出：目标域 4 类运动想象（左手 / 右手 / 双脚 / 舌头）的类别预测
评估：目标域测试集 classification accuracy + Cohen's Kappa（9 折跨被试平均）
```

核心目标：

### 理解部分

1. **理解域偏移（domain shift）**：为什么跨被试分类会失效？源域和目标域到底差在哪？
2. **理解 DANN 的三部分结构**：特征提取器 $G_f$、标签预测器 $G_y$、域判别器 $G_d$ 各司其职；
3. **理解对抗目标**：min-max 优化到底在「对抗」什么？为什么域判别器越强，特征越域不变？
4. **理解梯度反转层（GRL）**：前向是恒等映射、反向乘以 $-\lambda$，它是怎么把 min-max 变成一次反向传播的？
5. **理解 $\lambda$ 退火调度**：为什么域适配的权重要从 0 逐渐增大，而不是一开始就拉满？

### 实践部分

6. 复用之前复现过的 **EEGNet / EEGConformer** 作为特征提取器 $G_f$（去掉分类头）；
7. 实现完整的域对抗训练循环（源域有标签 + 目标域无标签的混合批训练）；
8. 记录并分析 label loss、domain loss、$\lambda$ 调度曲线，以及源域/目标域 acc 变化；
9. 用 **t-SNE** 可视化 $G_f$ 的特征分布，直观验证「域不变」是否达成；
10. 与 **source-only 基线**（不做域适配）对比，证明 DANN 确实带来了增益。

## 3. 数据集与域设置

使用公开数据集 **BCI Competition IV 2a**，采用**跨被试（leave-one-subject-out, LOSO）** 设置构造域适配任务。

- 数据集：9 名被试，4 类运动想象（左手、右手、双脚、舌头），22 导联，250 Hz。
- 官方下载：https://www.bbci.de/competition/iv/#dataset2a
- 也可使用 **MOABB** 库自动下载。

### 域划分

对每个被试 i（共 9 折，每个被试依次充当目标域）：

| 角色 | 组成 | 是否用标签训练 |
|:---:|:---:|:---:|
| 源域（Source） | 其余 8 名被试的训练集（8 × 288 = 2304 trials） | 使用 |
| 目标域-训练（Target train） | 被试 i 的训练集（288 trials） | 不用（无监督） |
| 目标域-测试（Target test） | 被试 i 的测试集（288 trials） | 仅最终评估用 |

要求：

- **严格的无监督域适配协议**：训练阶段只能接触源域标签 + 目标域训练数据（无标签）；目标域测试集只在最终评估时使用一次；
- 数据加载时要能同时产出 `(x, y, domain)`，其中 `domain=0` 表示源域、`domain=1` 表示目标域，目标域样本的 `y` 在损失计算时被忽略；
- 数据集文件不要提交到 Git 仓库（加入 `.gitignore`）。

## 4. 模型结构要求

需要以类的形式自行搭建 DANN，其中特征提取器 $G_f$ 要求**复用你自己之前实现的 EEGNet / EEGConformer backbone**（去掉最后的分类层），不能直接调用现成的 DANN pip 包。

### 4.1 DANN 架构概览

```text
Input EEG (C, T)
    │
    ▼
[Feature Extractor G_f]   —— EEGNet 或 EEGConformer（去掉分类头）
    │   输出特征向量 f = G_f(x) ∈ R^d（域不变特征）
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
[Label Predictor G_y]                       [Gradient Reversal Layer GRL]
    │  FC(d -> 256) -> ReLU                     │  前向：恒等 R(f)=f
    │  FC(256 -> 4)                             │  反向：×(-λ)
    ▼                                              ▼
L_y = CE(ŷ, y_s)      (仅源域)           [Domain Classifier G_d]
                                             │  FC(d -> 256) -> ReLU
                                             │  FC(256 -> 2)
                                             ▼
                                          L_d = CE(d̂, d)  (源+目标)
```

### 4.2 各组件详细说明

#### 4.2.1 特征提取器 $G_f$

$G_f$ 就是你在 EEGNet / EEGConformer 复现中写好的 backbone，只需**去掉最后的分类层**，让它输出一个特征向量 $f$：

- **EEGNet**：去掉最后的 `Linear(F2, num_classes)`，把 flatten 后的特征作为 $f$；
- **EEGConformer**：去掉 Classification Head 里的 `Linear(d_model, num_classes)`，把 Global Avg Pooling 之后的 $d_{model}$ 维向量作为 $f$。

**本次作业的看点之一**：对比这两种 backbone 在域适配下的表现——更强的特征提取器（EEGConformer）是否一定带来更好的跨被试泛化？这本身就是一个值得在报告中分析的实验。

#### 4.2.2 标签预测器 $G_y$

$G_y$ 是一个简单的多层感知机（MLP），把域不变特征映射到 4 个类别的 logits，用 softmax + 交叉熵计算分类损失 $L_y$。**$L_y$ 只在源域样本上计算**，因为目标域没有标签。

#### 4.2.3 域判别器 $G_d$

$G_d$ 也是一个 MLP，把特征 $f$ 映射到 2 类（源域 / 目标域），判断这个特征来自哪个域。它的任务是「尽可能分得清域」。

#### 4.2.4 梯度反转层（GRL）—— 本次作业的核心

GRL 是一个没有可学习参数的「伪层」，行为如下：

- **前向传播**：恒等映射，$R(f) = f$；
- **反向传播**：把传来的梯度乘以 $-\lambda$（$\lambda > 0$）。

它的作用：在更新 $G_f$ 时，把「域判别损失」的梯度**取反**，从而让 $G_f$ 朝「使 $G_d$ 分不清域」的方向更新；与此同时 $G_d$ 自身仍朝「分得清域」的方向更新。这样一个正向、一个反向，就构成了对抗。

你需要理解并回答：
- 为什么不能简单地把 $L_d$ 前面加个负号，而要用 GRL？
- GRL 的位置为什么必须在 $G_f$ 和 $G_d$ 之间？
- 去掉 GRL（或把 $\lambda$ 设为 0）会发生什么？

#### 4.2.5 $\lambda$ 退火调度

DANN 原文建议不要让域适配从一开始就全速进行，否则早期特征还没学好分类就会被「搅乱」。常见做法是让 $\lambda$ 随训练进度逐渐增大：

$$\lambda_p = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1, \quad p = \frac{\text{当前迭代}}{\text{总迭代}} \in [0,1],\ \gamma = 10$$

$p$ 从 0 线性增长到 1，$\lambda$ 从 0 逐渐增大并趋于 1。你需要在报告中画出这条调度曲线，并分析它对训练稳定性的影响。

## 5. 损失函数与训练目标

### 5.1 两个损失

- **标签损失（source only）**：$L_y = \text{CE}\big(G_y(G_f(x_s)),\ y_s\big)$
- **域判别损失（source + target）**：$L_d = \text{CE}\big(G_d\big(R(G_f(x))\big),\ d\big)$，其中 $d=0$（源）、$d=1$（目标）

### 5.2 对抗目标（min-max）

$$\min_{\theta_f,\theta_y}\ \max_{\theta_d}\ \Big[ L_y - \lambda \cdot L_d \Big]$$

- $G_d$（$\max$）想把 $L_d$ 降到最低，即「分得清域」；
- $G_f$（$\min$，通过 GRL 取反）想把 $L_d$ 升到最高，即「分不清域」；
- 同时 $G_f$ 和 $G_y$ 还要把 $L_y$ 降下来（分类要准）。

### 5.3 代码里的实现方式

因为 GRL 已经负责「取反」，min-max 目标可以简化成**一次反向传播**。常用的实现方式是：损失直接写为 $L = L_y + L_d$（两个损失都不额外加权），把 $\lambda$ 完全内置在 GRL 里（反向乘 $-\lambda$）。此时：

- $\theta_y$ 的梯度只来自 $L_y$，正常更新；
- $\theta_d$ 的梯度只来自 $L_d$，正常更新（朝分得清域）；
- $\theta_f$ 的梯度 = $\frac{\partial L_y}{\partial f} - \lambda \frac{\partial L_d}{\partial f}$，即「分类要准」且「分不清域」。

你需要理解：这个实现为什么等价于上面的 min-max 目标？平衡点在哪里？

## 6. 训练要求

### 6.1 基本训练设置

```text
dataset: BCI Competition IV 2a（跨被试 LOSO）
backbone: EEGNet / EEGConformer（两者都做，作为消融）
epochs: at least 100
batch size: 32 or 64（每个 batch 内源域、目标域各占一半）
optimizer: Adam / AdamW
learning rate: 1e-3 or 5e-4
label loss: CrossEntropyLoss（仅源域）
domain loss: CrossEntropyLoss（源域 + 目标域）
λ schedule: 见 4.2.5 的退火公式（γ=10），λ 内置在 GRL 中
evaluation: 目标域 accuracy + Kappa
```

### 6.2 训练循环要求

每个 epoch 需要记录：

- 源域分类 loss（$L_y$）与源域 accuracy
- 域判别 loss（$L_d$）与域判别 accuracy
- 目标域（训练部分）的 accuracy（用无标签数据 monitor，仅观察趋势）
- 当前 $\lambda$ 值

### 6.3 评估要求

训练结束后，在**目标域测试集**上报告：

- target test accuracy
- target test Kappa
- 混淆矩阵
- 9 折（9 个被试）的逐被试结果表 + 平均值

最低结果要求：

- 目标域测试 accuracy 应明显优于 random baseline（4 类 = 25%），一般可达 60%-75%；
- **必须与 source-only 基线对比**：不接域判别器（等价于 $\lambda=0$）训练同样的 backbone，DANN 应带来提升；
- 给出 **t-SNE 特征可视化**（源域 vs 目标域，适配前 vs 适配后）。

## 7. 最低完成标准

1. 能够加载 BCI IV 2a 并按 LOSO 构造 source / target 域，正确输出 `(x, y, domain)`；
2. 能够复用 EEGNet / EEGConformer 作为 $G_f$，自行搭建 $G_y$、$G_d$ 与 GRL；
3. 理解并能口头解释 GRL 的前向/反向行为，以及 DANN 的 min-max 目标；
4. 能够实现域对抗训练循环（含 $\lambda$ 退火），并打印每个 epoch 的 label loss、domain loss、源/目标 acc；
5. 能够在目标域测试集上评估并报告 accuracy、Kappa、混淆矩阵（9 折平均）；
6. 能够可视化 t-SNE 特征分布，并与 source-only 基线对比，分析域适配效果。

## 8. 最终提交内容

1. **实现代码**：数据加载（含域标签）、$G_f$（复用 backbone）、$G_y$、$G_d$、GRL、训练脚本、评估脚本；**GRL 与域对抗训练循环必须详细注释**。

2. **实验报告**：使用 `report-template.md` 模板撰写，包含任务说明、域设置、模型结构（重点展开 GRL 与对抗目标）、训练设置、结果分析、t-SNE 可视化与 source-only 对比。

3. **训练过程记录**：label loss / domain loss / $\lambda$ 调度曲线或关键日志截图，说明模型确实收敛、域判别确实被「对抗」住。

4. **特征可视化**：适配前后 $G_f$ 特征的 t-SNE 图（按域着色 + 按类着色），并分析域不变特征是否达成。

## 9. 过程记录与防作弊要求

为了确认作业是本人逐步完成，而非一次性生成成品，需满足以下过程性要求。

### Git 小步提交

每完成一个模块提交一次 commit，禁止一次性 push 整个项目。

合格 commit 粒度示例：

```text
feat: 加载 BCI IV 2a 并按 LOSO 构造 source/target 域（含 domain 标签）
feat: 复用 EEGNet 作为特征提取器 G_f（去掉分类头）
feat: 实现标签预测器 G_y
feat: 实现域判别器 G_d
feat: 实现梯度反转层 GRL（前向恒等、反向 ×(-λ)）
feat: 组合完整 DANN 模型
feat: 实现域对抗训练循环（含 λ 退火调度）
feat: 实现目标域评估（accuracy / Kappa / 混淆矩阵）
feat: 添加 t-SNE 特征可视化（源域 vs 目标域）
feat: 添加 source-only 基线对比
fix: 修复 GRL 反向传播梯度未取反的问题
docs: 补充实验报告中的 loss 曲线与 t-SNE 分析
```

提交时一并附上：

```text
仓库地址：GitHub / Gitee 均可
git log --oneline 的文本输出或截图
```

## 10. 参考资源

- **DANN 原始论文（本次目标方法）**：
  Ganin, Y., & Lempitsky, V. (2015). Domain-Adversarial Training of Neural Networks. 
  论文地址：https://arxiv.org/abs/1505.07818

- **MOABB 文档**（加载 BCI 数据集）：https://moabb.neurotechx.com/docs/index.html

- BCI Competition IV 2a 数据集：https://www.bbci.de/competition/iv
