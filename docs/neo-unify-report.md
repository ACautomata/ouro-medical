# NEO-unify 技术报告

> 来源: [HuggingFace Blog - SenseNova/neo-unify](https://huggingface.co/blog/sensenova/neo-unify) & [GitHub - OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)
> 
> 技术报告: [arXiv:2605.12500](https://arxiv.org/abs/2605.12500)

---

## 一、模型架构

### 1.1 核心设计理念

- **Encoder-Free Design**: 完全移除视觉编码器（VE）和变分自编码器（VAE）
- **Native Mixture-of-Transformer (MoT) 骨干网络**: 协同处理理解与生成任务
- **双分支架构**:
  - **Understanding Pathway (und)**: 语义理解分支（可冻结）
  - **Generation Pathway**: 像素级生成分支

### 1.2 输入输出接口

- **Near-lossless Visual Interface**: 输入输出均采用近无损视觉表示
- 表示空间由模型自身学习构建

### 1.3 统一学习范式

| 模态 | 训练方法 |
|------|----------|
| 文本 | 自回归交叉熵损失 (Autoregressive Cross-Entropy) |
| 图像 | 像素流匹配 (Pixel Flow Matching) |

---

## 二、训练细节

### 2.1 训练阶段

```
Web-scale Pretraining → Mid-training (MT) → Supervised Fine-tuning (SFT)
```

### 2.2 SFT训练流程

1. Understanding Warmup
2. Generation Pre-training
3. Unified Mid-training
4. Unified SFT
5. T2I RL训练（生成最终模型）

### 2.3 数据配置

- **预训练**: Web-scale 数据
- **中训练/微调**: 多样化高质量数据语料库
- **图像编辑训练**: 60k 混合训练步（使用公开 T2I 和编辑数据集）

### 2.4 联合训练策略

- **双分支联合训练**: 在相同的中训练和监督微调数据源上同时训练
- **低数据比例和损失权重**: 仍能保持理解分支稳定，生成分支收敛更快
- **协同进化**: 在 MoT 骨干网络中实现最小内在冲突

---

## 三、benchmark结果

### 3.1 图像重建性能（MS COCO 2017）

| 模型 | PSNR | SSIM |
|------|------|------|
| **NEO-unify (2B)** | **31.56** | **0.85** |
| Flux VAE | 32.65 | 0.91 |

> 注：NEO-unify 在仅需 90K 预训练步后达到上述性能

### 3.2 图像编辑性能

- **ImgEdit Score**: 3.32（冻结理解分支，仅 60k 混合训练）

---

## 四、关键技术特点

### 4.1 近无损输入优势

- 支持语义理解和像素级保真度兼顾
- 无需预训练编码器

### 4.2 数据扩展效率

- 相比 Bagel 等对手模型，NEO-unify 以更少训练 tokens 达到更高性能
- 在训练数据 scaling 方面表现显著优异

### 4.3 推理时冻结能力

- 编辑任务中可冻结理解分支，生成分支独立运作
- 大幅提升 token 利用效率

### 4.4 MoT (Mixture of Tokens) 架构

- 理解参数和生成参数可分离配置
- 通过原生 MoT 实现跨模态高效推理

---

## 五、模型系列

| 模型名称 | 参数规模 | 可用性 |
|---------|---------|--------|
| SenseNova-U1-8B-MoT | 8B MoT | HuggingFace |
| SenseNova-U1-A3B-MoT | A3B MoT | HuggingFace |
| SenseNova-U1-8B-MoT-SFT | 8B MoT | HuggingFace |
| SenseNova-U1-A3B-MoT-SFT | A3B MoT | HuggingFace |
| SenseNova-U1-8B-MoT-Infographic | 8B MoT | HuggingFace |

> 8B-MoT 指约8B理解参数和约8B生成参数

---

## 六、性能指标

### 6.1 生成速度

- H100/H200单节点：~0.15 s/step
- ~9s端到端生成2048×2048图像

### 6.2 推理优化

- 相比Triton基线提升2.4-3.2倍prefill速度

---

## 七、未来展望

- Interleaved perception–generation loops
- Omni-modal reasoning
- Vision-centric intelligence
- Spatial intelligence
- World model

---

## 八、相关资源

- **GitHub**: https://github.com/OpenSenseNova/SenseNova-U1
- **HuggingFace集合**: [sensenova/sensenova-u1](https://huggingface.co/collections/sensenova/sensenova-u1)
- **技术报告**: [arXiv:2605.12500](https://arxiv.org/abs/2605.12500)
- **许可证**: Apache 2.0
