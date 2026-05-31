# MeanFlow 深度调研报告

> 调研时间：2026-05-30 | 用途：为 mobius 项目（基于 Ouro Looped LM 的医学生成模型）提供 flow matching 加速方案参考

---

## 目录

1. [核心论文概览](#1-核心论文概览)
2. [数学框架](#2-数学框架meanflow-identity)
3. [系列演进：MF → iMF → pMF → Re-MF](#3-系列演进)
4. [代码仓库与实现细节](#4-代码仓库与实现)
5. [与相关方法的对比](#5-与相关方法的对比)
6. [在医学/3D领域的关联工作](#6-在医学3d领域的关联工作)
7. [对 mobius 项目的启发](#7-对-mobius-项目的启发)
8. [参考文献](#8-参考文献)

---

## 1. 核心论文概览

### MeanFlow (MF) — 原始论文

| 项目 | 内容 |
|------|------|
| **标题** | Mean Flows for One-step Generative Modeling |
| **arXiv** | [2505.13447](https://arxiv.org/abs/2505.13447) |
| **发表** | NeurIPS 2025 Oral |
| **作者** | Zhengyang Geng (CMU/MIT), Mingyang Deng (MIT), Xingjian Bai (MIT), J. Zico Kolter (CMU), **Kaiming He (MIT)** |
| **核心贡献** | 提出 **average velocity（平均速度）** 替代 instantaneous velocity（瞬时速度），实现 single-NFE 一步生成 |
| **关键结果** | ImageNet 256×256, **FID 3.43** (1-NFE, from scratch)，比之前 SOTA 相对提升 50-70% |
| **特点** | 完全从零训练，无需 pre-training、distillation、curriculum learning |

### Improved MeanFlow (iMF)

| 项目 | 内容 |
|------|------|
| **标题** | Improved Mean Flows: On the Challenges of Fastforward Generative Models |
| **arXiv** | [2512.02012](https://arxiv.org/abs/2512.02012) |
| **作者** | Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He (CMU/MIT/Adobe/THU) |
| **核心改进** | (1) 将 u-loss 重参数化为 **v-loss**，消除训练目标对网络自身的依赖；(2) **灵活 CFG**：将 guidance scale 作为条件变量而非固定训练；(3) **in-context conditioning** 替代 adaLN-zero，模型体积缩小 1/3 |
| **关键结果** | ImageNet 256×256, **FID 1.72** (1-NFE, from scratch)，相对 MF 再提升 50% |

### Pixel MeanFlow (pMF)

| 项目 | 内容 |
|------|------|
| **标题** | One-step Latent-free Image Generation with Pixel Mean Flows |
| **arXiv** | [2601.22158](https://arxiv.org/abs/2601.22158) |
| **作者** | Yiyang Lu, Susie Lu, Qiao Sun, ..., Zhengyang Geng, Kaiming He |
| **核心思想** | 将 **预测空间** (x-prediction) 和 **损失空间** (v-loss) 分离，实现 pixel-space 端到端一步生成（无需 VAE latent） |
| **转换链** | `x → u → v`：网络输出 x-prediction，通过 MeanFlow identity 转换为 v-loss |
| **关键结果** | ImageNet 256×256: **FID 2.22** (1-NFE, pixel-space)；512×512: **FID 2.48** |

### Rectified MeanFlow (Re-MF)

| 项目 | 内容 |
|------|------|
| **标题** | Flow Straighter and Faster: Efficient One-Step Generative Modeling via MeanFlow on Rectified Trajectories |
| **arXiv** | [2511.23342](https://arxiv.org/abs/2511.23342) |
| **作者** | Xinxi Zhang, Shiwei Tan 等 (Rutgers) |
| **核心思想** | 结合 Rectified Flow 的 trajectory straightening + MeanFlow 的 mean-velocity learning；仅用 **1 次 reflow** 而非多次，大幅降低训练开销 |
| **关键结果** | ImageNet 64×64: FID 2.87 (比 2-rectified flow++ 提升 33.4%，训练成本仅 10%)；512×512: FID 3.03 |

---

## 2. 数学框架：MeanFlow Identity

### 2.1 背景：Flow Matching 回顾

Flow Matching (Lipman et al., 2023) 学习一个 **瞬时速度场** v(x_t, t)，通过 ODE 积分从噪声生成数据：

```
dx_t/dt = v(x_t, t),  x_1 = ε ~ N(0, I)  →  x_0 ~ p_data
```

采样需要多步 ODE 求解（如 Euler 方法），通常需要 50-1000 步。

### 2.2 MeanFlow 核心创新：Average Velocity

MeanFlow 引入 **平均速度场** u(z_t, r, t)，定义为时间区间 [r, t] 内瞬时速度的积分平均值：

```
u(z_t, r, t) = (1/(t-r)) ∫_r^t v(z_τ, τ) dτ         ... (Eq. 3)
```

**关键优势**：如果准确学习 u，则可以从噪声 ε **一步** 生成数据：

```
x_0 ≈ ε - (1-0) · u_θ(ε, 0, 1) = ε - u_θ(ε, 0, 1)   ... 一步生成！
```

### 2.3 MeanFlow Identity（核心等式）

平均速度 u 与瞬时速度 v 之间存在一个 **恒等关系**（非人为引入的约束）：

```
v(z_t, t) = u(z_t, r, t) + (t - r) · d/dt u(z_t, r,t)   ... (Eq. 6)
```

其中全导数：
```
d/dt u = ∂_t u + (∇_x u) · v(z_t, t)                     ... (Eq. 7)
```

**重要性质**：
- 这个 identity 是 **定义导出的**，不依赖神经网络假设
- 无需额外的 consistency 约束或 curriculum learning
- 训练目标天然存在 ground-truth target field

### 2.4 训练损失函数

#### 原始 MF（u-loss）

网络 u_θ 学习满足 MeanFlow Identity，损失为：

```
L = E[||u_θ(z_t, r, t) - target||²]
```

其中 target 依赖于网络自身的 JVP（Jacobian-vector product），这是 iMF 要解决的问题。

#### 改进 iMF（v-loss 重参数化）

iMF 将损失重新表述为 **对 v 的回归**，用 u_θ 重参数化：

```
V_θ = u_θ + (t-r) · d/dt u_θ     （通过 JVP 计算全导数）
L = E[||V_θ - v||²]               （v 是 ground-truth 瞬时速度，已知）
```

**好处**：损失目标 v 是 **不依赖网络的固定量**，形成标准回归问题，训练更稳定。

#### pMF（x-prediction + v-loss）

pMF 网络直接输出 x-prediction，通过转换链计算 loss：

```
网络输出: x_θ(z_t, r, t)
转换:     u = (z_t - x_θ) / t        ... (Eq. 8 逆)
          V_θ = u + (t-r) · d/dt u
Loss:     ||V_θ - v||²
```

### 2.5 Classifier-Free Guidance (CFG) 集成

MF 的独特之处在于 CFG 可以 **训练时内置** 到 target field 中：

- **原始 MF**：固定 guidance scale w，训练时直接学习 `u^cfg = (1+w)·u_cond - w·u_uncond`
- **iMF**：将 w 作为 **条件变量**，保留推理时的灵活性（仍为 1-NFE）

### 2.6 采样算法

```
# MeanFlow 一步采样 (Algorithm 1)
Input: noise ε ~ N(0, I), class label c
Output: generated sample x_0

u = u_θ(ε, r=0, t=1, c)    # 单次前向传播
x_0 = ε - u                 # 一步到位

# MeanFlow 两步采样 (可提升质量)
u1 = u_θ(ε, r=0, t=0.5, c)
z_half = ε - 0.5 * u1
u2 = u_θ(z_half, r=0.5, t=1, c)
x_0 = z_half - 0.5 * u2
```

---

## 3. 系列演进

### 性能对比（ImageNet 256×256, 1-NFE）

| 方法 | 参数量 | FID | 训练方式 |
|------|--------|-----|----------|
| iCT-XL/2 | 675M | 34.24 | From scratch |
| Shortcut-XL/2 | 675M | 10.60 | From scratch |
| MeanFlow-B/2 | 131M | 6.17 | From scratch |
| MeanFlow-L/2 | 459M | 3.84 | From scratch |
| **MeanFlow-XL/2** | **676M** | **3.43** | **From scratch** |
| Re-MeanFlow (from SiT-XL) | - | 3.41 | Distillation (1 reflow) |
| **iMF-XL/2** | **~676M** | **1.72** | **From scratch** |
| iMF-XL/2 (2-NFE) | ~676M | 1.54 | From scratch |

### 各版本改进要点

```
MF (v1)                         iMF (v2)                     pMF (v3)
┌─────────────────┐    ┌─────────────────────┐    ┌────────────────────┐
│ u-loss          │    │ v-loss (重参数化)    │    │ x-prediction       │
│ 固定 CFG scale  │ →  │ 灵活 CFG (条件变量)  │ →  │ + v-loss           │
│ adaLN-zero      │    │ in-context cond.     │    │ 无 VAE latent      │
│ FID 3.43        │    │ FID 1.72             │    │ 端到端 pixel-space │
│                 │    │ 模型体积 -33%        │    │ FID 2.22 (256)     │
└─────────────────┘    └─────────────────────┘    └────────────────────┘
                                                      ↓
                                               Re-MF (变体)
                                               ┌────────────────────┐
                                               │ Rectified Flow +   │
                                               │ MeanFlow           │
                                               │ 仅 1 次 reflow     │
                                               │ 曲率瓶颈缓解       │
                                               │ 训练成本 -90%      │
                                               └────────────────────┘
```

---

## 4. 代码仓库与实现

### 4.1 官方仓库

| 仓库 | 语言 | 链接 |
|------|------|------|
| **Gsunshine/meanflow** | JAX (官方) | https://github.com/Gsunshine/meanflow |
| **Gsunshine/meanflow** (PyTorch branch) | PyTorch | 同上，含 PyTorch 分支 |
| **Lyy-iiis/imeanflow** | JAX + PyTorch | https://github.com/Lyy-iiis/imeanflow (iMF + pMF) |
| **zhuyu-cs/MeanFlow** | PyTorch (社区) | https://github.com/zhuyu-cs/MeanFlow |
| **Xinxi-Zhang/Re-MeanFlow** | PyTorch | https://github.com/Xinxi-Zhang/Re-MeanFlow |

### 4.2 官方代码结构 (Gsunshine/meanflow)

```
meanflow/
├── configs/          # 训练配置 (B/2, L/2, XL/2 等)
├── models/
│   ├── meanflow.py   # MeanFlow 核心模型
│   ├── dit.py        # DiT backbone (Transformer)
│   └── ...
├── train.py          # JAX + TPU 训练脚本
├── sample.py         # 采样 + FID 评估
├── ema.py            # EMA 权重更新
└── install.sh        # 依赖安装
```

### 4.3 关键实现细节

**Backbone 架构**：基于 DiT (Diffusion Transformer)
- 使用预训练 VAE tokenizer（SD-VAE）：256×256 → 32×32×4 latent
- MF/iMF: adaLN-zero 或 in-context conditioning
- pMF: 纯 Transformer，无 latent space

**训练配置**（MF-XL/2 为例）：
- 参数量：676M (XL/2)
- 训练硬件：TPU
- EMA rate: 0.9999
- Batch size: 大规模 (512+)
- 训练迭代：~1M steps

**JVP 计算**（关键代码路径）：
- MeanFlow Identity 的 `d/dt u_θ` 需要 Jacobian-vector product
- iMF 使用 autograd-based JVP 实现
- 这是训练中最昂贵的操作之一

### 4.4 社区 PyTorch 实现 (zhuyu-cs/MeanFlow)

```
MeanFlow/
├── meanflow/
│   ├── meanflow_model.py     # MeanFlow 核心逻辑
│   ├── unet.py               # U-Net backbone
│   └── ...
├── train.py                  # 训练入口
├── sample.py                 # 采样
└── README.md
```

注意：社区实现基于 U-Net backbone，且需要 **预训练 flow matching 模型** 作为起点。

---

## 5. 与相关方法的对比

### 5.1 方法论对比

| 方法 | 核心思想 | 训练条件变量 | 需要蒸馏 | 需要课程学习 |
|------|----------|-------------|----------|-------------|
| **Consistency Models (CM)** | 数据端锚定的 self-consistency | 单时间 t | 可选 | 是（discretization） |
| **Consistency Distillation (CD)** | 从预训练扩散模型蒸馏 | 单时间 t | 是 | 是 |
| **Rectified Flow** | Reflow 使轨迹变直 | 单时间 t | 否 | 否（但需多次 reflow） |
| **Shortcut Models** | 两时间自一致性约束 | 双时间 (r,t) | 否 | 否 |
| **IMM** | 两时间 self-consistency | 双时间 (r,t) | 否 | 部分 |
| **MeanFlow (MF)** | **平均速度场的定义恒等式** | **双时间 (r,t)** | **否** | **否** |

### 5.2 MeanFlow 的独特优势

1. **原理性**：Identity 是从平均速度定义自然导出的，不是人为约束
2. **存在 ground-truth**：target field u 不依赖网络，最优解理论上独立于网络架构
3. **无需 distillation**：完全从零训练达到 SOTA
4. **CFG 内置**：guidance 可以融入 target field，不增加推理开销
5. **可扩展**：支持 1-NFE 到 few-NFE 的灵活推理

### 5.3 局限性

1. **JVP 开销**：训练需要 Jacobian-vector product，计算昂贵
2. **曲率瓶颈**（Re-MF 指出）：在高度弯曲的 flow 上训练收敛慢
3. **仅验证图像生成**：尚未在 3D 医学影像领域验证
4. **TPU-first**：官方代码以 JAX/TPU 为主，PyTorch 支持较晚

---

## 6. 在医学/3D领域的关联工作

### 6.1 Flow Matching 在医学影像中的应用

| 工作 | 方法 | 应用 | 关键点 |
|------|------|------|--------|
| **MedFlowSeg** (2026) | Flow Matching + 双分支注意力 | 医学图像分割 | 将分割建模为 flow matching 的生成过程，单步推理 |
| **MOTFM** (2025) | Optimal Transport FM | 医学图像合成(2D/3D) | 支持 class/mask 条件、2D/3D、10 步采样 |
| **AlignFlow** (2026) | Few-shot 分布对齐 FM | 医学图像分割数据增强 | 用少量参考图像对齐生成分布 |

### 6.2 与 mobius 项目的关联

mobius 项目当前使用的是基于 Ouro Looped LM 的 **标准 Flow Matching**（instantaneous velocity），50 步 Euler ODE 推理。MeanFlow 的启发：

1. **推理加速**：MeanFlow 可将 50 步 ODE 降为 1 步，对 3D 医学体积数据意义重大
2. **训练稳定性**：iMF 的 v-loss 重参数化思路可借鉴，避免训练目标依赖网络
3. **3D 适配挑战**：
   - MeanFlow Identity 的 JVP 在 3D 体积上计算量更大
   - 3D 的 flow 曲率问题更严重（Re-MF 的 rectification 可能必要）
   - pMF 的 x-prediction 思路可能更适合医学影像（manifold 假设）

---

## 7. 对 mobius 项目的启发

### 7.1 可直接借鉴的技术

| 技术 | 来源 | 适用场景 |
|------|------|----------|
| v-loss 重参数化 | iMF | 替代当前 FM 的 v-prediction loss |
| in-context conditioning | iMF | 多条件（subtype、modality）注入 |
| x-prediction + v-loss | pMF | 医学影像的 manifold 友好预测 |
| Trajectory rectification | Re-MF | 降低 3D flow 曲率，加速训练 |
| 灵活 CFG | iMF | 条件生成的推理时可控性 |

### 7.2 潜在研究方向

1. **3D MeanFlow**：将 MeanFlow Identity 扩展到 3D 体积数据（BraTS2023 MRI）
2. **MeanFlow + Looped LM**：用 Ouro 的循环推理能力建模 mean velocity field
3. **医学特定的曲率优化**：3D 医学数据的 flow 曲率分析与 rectification
4. **x-prediction 在医学影像中的优势**：医学影像的 manifold 结构可能比自然图像更规则

### 7.3 技术风险评估

| 风险 | 级别 | 缓解方案 |
|------|------|----------|
| 3D JVP 计算爆炸 | 高 | 使用 Re-MF 的 rectified trajectory 降低曲率 |
| 3D 体积的 latent space | 中 | pMF 的 pixel-space 方案或 3D VAE |
| 训练稳定性 | 中 | iMF 的 v-loss + gradient modulation |
| 医学影像特定损失 | 低 | 保留现有的 flow matching loss 作为 baseline |

---

## 8. 参考文献

### 核心论文

1. **Geng, Z., Deng, M., Bai, X., Kolter, J.Z., & He, K.** (2025). Mean Flows for One-step Generative Modeling. NeurIPS 2025 Oral. [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)

2. **Geng, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J.Z., & He, K.** (2025). Improved Mean Flows: On the Challenges of Fastforward Generative Models. [arXiv:2512.02012](https://arxiv.org/abs/2512.02012)

3. **Lu, Y., Lu, S., Sun, Q., ..., Geng, Z., & He, K.** (2026). One-step Latent-free Image Generation with Pixel Mean Flows. [arXiv:2601.22158](https://arxiv.org/abs/2601.22158)

4. **Zhang, X., Tan, S., et al.** (2026). Overcoming the Curvature Bottleneck in MeanFlow (Re-MeanFlow). [arXiv:2511.23342](https://arxiv.org/abs/2511.23342)

5. **You, H., Liu, B., & He, H.** (2025). Modular MeanFlow: Towards Stable and Scalable One-Step Generative Modeling. [arXiv:2508.17426](https://arxiv.org/abs/2508.17426)

### 背景方法

6. **Lipman, Y., et al.** (2023). Flow Matching for Generative Modeling. ICLR 2023.
7. **Lipman, Y., et al.** (2024). Flow Matching Guide and Code. [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
8. **Song, Y., et al.** (2023). Consistency Models. ICML 2023.
9. **Liu, X., et al.** (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023.

### 医学领域关联

10. **MedFlowSeg** (2026). Flow Matching for Medical Image Segmentation with Frequency-Aware Attention. [arXiv:2604.19675](https://arxiv.org/abs/2604.19675)
11. **MOTFM** (2025). Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality. [arXiv:2503.00266](https://arxiv.org/abs/2503.00266)
12. **AlignFlow** (2026). Few-Shot Distribution-Aligned Flow Matching for Data Synthesis in Medical Image Segmentation. [arXiv:2604.02868](https://arxiv.org/abs/2604.02868)

### 代码仓库

- https://github.com/Gsunshine/meanflow (官方 JAX/PyTorch)
- https://github.com/Lyy-iiis/imeanflow (iMF + pMF 官方)
- https://github.com/zhuyu-cs/MeanFlow (社区 PyTorch)
- https://github.com/Xinxi-Zhang/Re-MeanFlow (Re-MF)
