# Ouro: Scaling Latent Reasoning via Looped Language Models — 完整信息汇总

> 来源整合自：[项目主页](https://ouro-llm.github.io/)、[arXiv 论文](https://arxiv.org/abs/2510.25741)、[HuggingFace 模型页面](https://huggingface.co/ByteDance/Ouro-1.4B)

---

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| **论文标题** | Scaling Latent Reasoning via Looped Language Models |
| **项目名称** | Ouro（取自衔尾蛇 Ouroboros，象征递归循环） |
| **arXiv 编号** | 2510.25741（v4, 2025年11月17日最后更新） |
| **开源协议** | Apache-2.0 |
| **代码仓库** | 官方声明 Coming Soon |
| **核心作者** | Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li 等（约30+人） |
| **通讯作者** | ridger@ucsc.edu, zhangge.eli@bytedance.com, huang.wenhao@bytedance.com, jsn@ucsc.edu |

### 机构

1. ByteDance Seed
2. UC Santa Cruz
3. Princeton University
4. Mila - Quebec AI Institute
5. University of Montreal
6. Peking University
7. Carnegie Mellon University
8. University of Pennsylvania
9. Conscium
10. University of Manchester
11. M-A-P

---

## 2. 核心动机与问题

现代 LLM 主要通过显式文本生成（如 Chain-of-Thought）来"思考"，将推理推迟到后训练阶段，未能充分利用预训练数据。Ouro 探索**第三条路径**——通过架构创新，在固定参数预算内实现动态计算：

1. **缩放模型大小** — 传统路线，受部署成本限制
2. **扩展推理时计算（CoT）** — 增加 token 生成长度，导致上下文膨胀
3. **循环架构（Ouro 的路线）** — 递归复用共享参数，在潜空间中进行迭代推理

### 核心研究问题

- LoopLM 在能力、效率和安全性方面是否比非循环 Transformer 有更好的缩放行为？
- 递归复用权重是否能获得与增加深度相同的能力增益？
- 循环的收益是否随循环次数单调递增？影响因素是什么？

---

## 3. 架构设计

### 3.1 Looped Language Model (LoopLM) 核心思想

将 N 层 Transformer block 重复执行 T 次前向传播（循环），共享参数：

```
F^(t)(·) = lmhead ∘ (M^L ∘ M^L ∘ ... ∘ M^L) ∘ emb(·)
                    \_____ t 次迭代 _____/
```

- t=1 时退化为普通 Transformer
- t=4（默认）为 Ouro 使用的配置

### 3.2 自适应计算 — Exit Gate 机制

每个循环步骤 t，一个 Exit Gate 预测退出概率：

- 瞬时退出概率：λ_t(x) = σ(Linear_φ(h^(t)))
- 生存概率：S_t(x) = ∏(1 - λ_j(x))
- 退出分布：p_φ(t|x) 构成有效离散分布

推理时使用 Q-exit 准则：当累积退出概率 CDF(t) 超过阈值 q 时退出。

### 3.3 熵正则化训练目标

总训练损失结合语言模型损失和 KL 散度正则化：

```
L_total = Σ_t p_φ(t|x) · L^(t) + β · KL(p_φ(·|x) || Uniform)
```

- β 控制 KL 惩罚强度（Stage 1a: 0.1，后期: 0.05）
- 均匀先验确保无偏的深度探索，防止 collapse 到浅层或过度使用长循环

### 3.4 自适应退出专项训练

在标准预训练后，增加一个专项训练阶段（冻结 backbone，仅训练 gate）：

```
L_adaptive = Σ_t p_φ(t|x) · [L^(t) + α · (-I_t^(n))]
```

- I_t^(n) = max(0, L_i,stop^(t-1) - L_i,stop^(t))：从步骤 n 到步骤 t 的损失改善
- 理想继续概率：w_i^(t) = sigmoid(k · (I_i^(t) - γ))
- 超参数：k = 50.0（斜率），γ = 0.005（阈值）

### 3.5 Sandwich Normalization

为稳定深层循环计算的训练，使用 Sandwich RMSNorm（每个解码器层包含 **4 个 RMSNorm**，标准 LLaMA 只有 2 个）：

```
标准 LLaMA (Pre-Norm):
  residual = x
  x = RMSNorm(x)             ← input_layernorm
  x = Attention(x)
  x = residual + x
  residual = x
  x = RMSNorm(x)             ← post_attention_layernorm
  x = MLP(x)
  x = residual + x

Ouro (Sandwich Norm):
  residual = x
  x = RMSNorm(x)             ← input_layernorm        (pre-attention)
  x = Attention(x)
  x = RMSNorm(x)             ← input_layernorm_2      (post-attention)  ★ 新增
  x = residual + x
  residual = x
  x = RMSNorm(x)             ← post_attention_layernorm (pre-MLP)
  x = MLP(x)
  x = RMSNorm(x)             ← post_attention_layernorm_2 (post-MLP)  ★ 新增
  x = residual + x
```

在循环架构中，同一组权重被反复执行，中间表示的范数容易失控。Post-norm 有助于在每步循环中稳定数值。

---

## 4. 模型规格

### Ouro-1.4B

| 配置 | 值 |
|---|---|
| 参数量 | 1.4B |
| 物理层数 | 24 |
| 循环步数 | 4（可调） |
| 等效总层数 | 24 × 4 = 96 |
| 隐藏维度 | 2048 |
| MLP 中间层维度 | 5632（约 2.75× hidden） |
| 注意力头数 | 16（MHA，非 GQA） |
| KV 头数 | 16 |
| 每头维度 | 128 |
| FFN 激活 | SwiGLU |
| 位置编码 | RoPE（theta = 1,000,000） |
| 词表大小 | 49,152 |
| 最大位置编码 | 65,536 |
| 训练上下文长度 | 4K → 扩展至 64K |
| 归一化 | Sandwich RMSNorm（4 个/层） |
| 权重初始化 | 正态分布 σ=0.02 |

### Ouro-2.6B

| 配置 | 值 |
|---|---|
| 参数量 | 2.6B |
| 物理层数 | 48（从 24 层 upcycling 而来） |
| 循环步数 | 4 |
| 等效总层数 | 48 × 4 = 192 |
| 其他配置 | 与 1.4B 相同 |

### 模型变体

- **Ouro-1.4B** — 基础预训练模型
- **Ouro-2.6B** — 基础预训练模型
- **Ouro-1.4B-Thinking** — SFT 推理增强版本
- **Ouro-2.6B-Thinking** — SFT 推理增强版本

---

## 5. 完整训练流程

### 总训练规模：7.7T tokens（base models）

### 全局优化器配置（所有阶段通用）

| 配置 | 值 |
|---|---|
| 优化器 | AdamW |
| weight decay | 0.1 |
| β₁ | 0.9 |
| β₂ | 0.95 |
| 梯度裁剪 | 1.0 |
| 学习率策略 | WSD (Warmup-Stable-Decay) |
| 训练框架 | flame（基于 torchtitan） |

### Stage 1a: 预训练 Phase I（探索阶段）

| 配置项 | 值 |
|---|---|
| 循环步数 | **8**（初始） |
| 学习率 | Warmup-Stable，峰值 **3 × 10⁻⁴** |
| 序列长度 | **4K** tokens |
| 初始 batch size | **4M** tokens → 渐增至 **8M** |
| KL 系数 β | **0.1** |
| 数据量 | **~3T** tokens |
| 主要数据 | Nemotron-CC (6.3T) 为主，辅以 Ultra-FineWeb-zh、MAP-CC、OpenCoder、MegaMath |

**发现问题**：8 步循环导致 loss spike 和梯度振荡，推测由多次循环的复合梯度流放大扰动所致。

### Stage 1b: 预训练 Phase II（稳定化 + Upcycling）

| 配置项 | 值 |
|---|---|
| 循环步数 | 从 8 **降至 4** |
| batch size | **8M** tokens |
| KL 系数 β | **0.1** |
| 数据量 | **~3T** tokens |

**分支策略（Model Branching）**：
- **Ouro-1.4B**：保留原始 24 层预训练权重 + 4 循环步 → 等效深度 96
- **Ouro-2.6B**：24 层通过 **layer duplication** 扩展到 48 层 + 4 循环步 → 等效深度 192

Upcycling 细节：将 24 层的每一层复制一份形成 48 层。由于 LoopLM 的循环架构中权重本身就在迭代间共享，这种层复制特别平滑，不会出现标准 Transformer upcycling 中的典型不稳定性。

### Stage 2: CT Annealing（持续训练退火）

| 配置项 | 值 |
|---|---|
| 学习率 | 退火至 **3 × 10⁻⁵** |
| 循环步数 | **4** |
| 序列长度 | **16K** tokens |
| batch size | **8M** tokens |
| KL 系数 β | **0.05**（从 0.1 降低） |
| 数据量 | **1.4T** tokens |

数据组成（精确比例）：

| 数据源 | 比例 |
|---|---|
| Nemotron-CC-high-quality | 66.5% |
| Nemotron-CC-Math-v1 | 15.0% |
| MegaMath-high-quality | 4.6% |
| OpenCoder-LLM/opc-annealing-corpus | 0.5% |
| Nemotron-pre-training-Code-v1/Synthetic-Code | 3.8% |
| Nemotron-pre-training-SFT-v1/Nemotron-SFT-Code | 3.4% |
| Nemotron-pre-training-SFT-v1/Nemotron-SFT-General | 6.2% |

### Stage 3: LongCT（长上下文训练）

| 配置项 | 值 |
|---|---|
| batch size | **8M** tokens |
| KL 系数 β | **0.05** |
| 序列长度 | **64K** tokens |
| 数据集 | ProLong 64K 子集 |
| 数据量 | **20B** tokens |

### Stage 4: Mid-Training（中期训练）

| 配置项 | 值 |
|---|---|
| 学习率 | **1 × 10⁻⁵**，cosine scheduler |
| 序列长度 | **32K** tokens |
| 数据量 | **300B** tokens（182B 处理后采样 90B + Stage 1 重放 30B + Stage 2 重放 180B） |

数据处理：
- 20+ 开源 SFT 数据集整合，全面去污染
- 包含 `<Question, Answer>` 和 `<Question, CoT, Answer>` 格式
- 全部转换为 **ChatML** 格式以减少后续 alignment tax

### KL 系数 β 变化总结

| 阶段 | β |
|---|---|
| Stage 1a | 0.1 |
| Stage 1b | 0.1 |
| Stage 2 | 0.05 |
| Stage 3 | 0.05 |
| Stage 4 | 0.05 |

降低 β 的双重目的：(1) 减少任务损失与 KL 惩罚间的冲突梯度，(2) 允许模型更大自由度探索有益深度模式。

### 序列长度递进

| 阶段 | 序列长度 |
|---|---|
| Stage 1 (Pre-training) | 4K |
| Stage 2 (CT Annealing) | 16K |
| Stage 3 (LongCT) | 64K |
| Stage 4 (Mid-Training) | 32K |

### Stage 1 预训练数据（Table 4，总 6T tokens）

| 数据源 | 说明 |
|---|---|
| Nemotron-CC (6.3T) | 主要 CommonCrawl 数据 |
| Ultra-FineWeb-zh | 中文数据（仅 Stage 1 使用） |
| MAP-CC | 中文数据（仅 Stage 1 使用） |
| OpenCoder | 代码数据 |
| MegaMath | 数学数据 |

注意：由于 tokenizer 无中文字符，中文在 Stage 2 之后被移除。

### SFT: Supervised Fine-Tuning

| 配置项 | 值 |
|---|---|
| 训练轮次 | 2 epochs |
| 最大序列长度 | 32K tokens |
| 训练框架 | LlamaFactory |
| 优化器 | Adam |
| 学习率 | 2 × 10⁻⁵ |
| β (Adam) | (0.9, 0.95) |
| 学习率策略 | cosine decay |

数据组成（总计约 8.3M 样本）：

| 主题 | 数据源 | 样本数 |
|---|---|---|
| 数学 | OpenThoughts3, AceReason-1.1-SFT | 3.5M |
| 代码 | AceReason-1.1-SFT, OpenCodeReasoning, Llama-Nemotron-Post-Training-Dataset, OpenThoughts3 | 3.2M |
| 科学 | OpenThoughts3, Llama-Nemotron-Post-Training-Dataset | 808K |
| 对话 | OO1-Chat-747K, DeepWriting-20K | 767K |

**特殊说明**：训练因基础设施问题中断，从上次保存的 checkpoint 恢复，学习率接近原始 cosine decay 计划。

---

## 6. 强化学习尝试（未成功）

SFT 后尝试了 RLVR 对齐（DAPO 和 GRPO on DAPO-17K），但未获得显著增益：

1. **Off-policy rollouts**：在 vLLM 中生成完整 4 步 rollouts，模拟提前退出。off-policy 不匹配导致无效。
2. **固定 4 轮 RL**：训练正常进行但性能未超越 SFT checkpoint。可能原因：小模型经大量 SFT 后 RL 空间有限。

**关键障碍**：vLLM/SGLang 的固定执行路径与 LoopLM 的动态深度计算不兼容。

---

## 7. 代码实现详细分析

> 本节基于 HuggingFace 上的 `modeling_ouro.py`、`configuration_ouro.py`、`config.json`、`tokenizer_config.json` 等文件进行源码级分析。

### 7.1 类继承结构

```
PreTrainedModel (HuggingFace)
  └── OuroPreTrainedModel
        ├── OuroModel                   (核心模型，执行 UT 循环 + Exit Gate)
        │     └── 内含: OuroDecoderLayer × N, OuroRotaryEmbedding, early_exit_gate
        ├── OuroForCausalLM             (语言模型头，多策略推理)
        │     └── mixins: GenerationMixin
        ├── OuroForSequenceClassification
        ├── OuroForTokenClassification
        └── OuroForQuestionAnswering

PretrainedConfig
  └── OuroConfig (含 total_ut_steps, early_exit_threshold 等)

Cache (HuggingFace)
  └── UniversalTransformerCache (UT 专用 KV 缓存，二维索引)

nn.Module 层级:
  OuroDecoderLayer
    ├── OuroAttention (q/k/v/o_proj + RoPE)
    ├── OuroMLP (SwiGLU: gate_proj, up_proj, down_proj)
    ├── input_layernorm          (pre-attention RMSNorm)
    ├── input_layernorm_2        (post-attention RMSNorm)    ← Sandwich Norm
    ├── post_attention_layernorm (pre-MLP RMSNorm)
    └── post_attention_layernorm_2 (post-MLP RMSNorm)        ← Sandwich Norm
```

**代码来源**：代码骨架源自 Qwen 系列模型（`configuration_ouro.py` 头部有 Qwen/Alibaba 版权声明），在其基础上添加了 UT 循环机制。

### 7.2 OuroConfig 配置参数

#### 标准 Transformer 参数

| 参数 | 1.4B 值 | 2.6B 值 | 默认值 | 说明 |
|---|---|---|---|---|
| `vocab_size` | 49,152 | 49,152 | 151,936 | 词表大小 |
| `hidden_size` | 2,048 | 2,048 | 4,096 | 隐藏层维度 |
| `intermediate_size` | 5,632 | 5,632 | 22,016 | MLP 中间层维度 |
| `num_hidden_layers` | 24 | 48 | 32 | **物理层数**（非等效总层数） |
| `num_attention_heads` | 16 | 16 | 32 | 注意力头数 |
| `num_key_value_heads` | 16 | 16 | 32 | KV 头数（MHA） |
| `head_dim` | 128 | 128 | — | 每头维度 |
| `hidden_act` | "silu" | "silu" | "silu" | 激活函数 |
| `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 | RMSNorm epsilon |
| `initializer_range` | 0.02 | 0.02 | 0.02 | 权重初始化标准差 |
| `max_position_embeddings` | 65,536 | 65,536 | 32,768 | 最大位置编码长度 |
| `rope_theta` | 1,000,000 | 1,000,000 | 10,000 | RoPE 基频（极大值） |
| `tie_word_embeddings` | false | false | false | 是否共享嵌入权重 |

#### Ouro 特有参数

| 参数 | 值 | 说明 |
|---|---|---|
| **`total_ut_steps`** | 4 | **UT 循环步数**——核心参数，决定每层的重复次数 |
| **`early_exit_threshold`** | 1.0 | **early exit 累积概率阈值**（1.0 = 不提前退出） |
| `use_sliding_window` | false | 是否使用滑动窗口注意力 |
| `layer_types` | ["full_attention"] × 24/48 | 每层的注意力类型列表 |

#### auto_map 自定义映射

```json
"auto_map": {
    "AutoConfig": "configuration_ouro.OuroConfig",
    "AutoModel": "modeling_ouro.OuroModel",
    "AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"
}
```

所有组件通过 auto_map 指向自定义实现，要求 `transformers >= 4.55.0`。

### 7.3 前向传播完整流程

#### OuroModel.forward — 核心循环引擎

```
输入: input_ids [batch, seq_len]
  │
  ├── embed_tokens(input_ids) → inputs_embeds [batch, seq_len, 2048]
  │
  ├── 初始化 UniversalTransformerCache(max_cache_size = 24 * 4 = 96)
  │     如果传入的 cache 不兼容，自动创建
  │
  ├── 计算 cache_position (基于已缓存 token 数)
  ├── 计算 position_ids = cache_position.unsqueeze(0)
  │
  ├── rotary_emb(inputs_embeds, position_ids) → (cos, sin)
  │     位置编码只计算一次，所有 UT 步共享
  │
  ├── 创建 attention_mask (causal_mask)
  │
  └── ══════ UT 循环（核心）══════
      │
      │  hidden_states_list = []
      │  gate_list = []
      │
      │  for current_ut in range(4):           ← 4 步循环
      │    │
      │    │  for layer in layers[:24]:         ← 遍历 24 个物理层
      │    │    │
      │    │    │  residual = hidden_states
      │    │    │  hidden_states = RMSNorm(hidden_states)
      │    │    │  hidden_states = Attention(hidden_states,
      │    │    │      cache_idx = ut * 24 + layer_idx)
      │    │    │  hidden_states = RMSNorm(hidden_states)
      │    │    │  hidden_states = residual + hidden_states
      │    │    │
      │    │    │  residual = hidden_states
      │    │    │  hidden_states = RMSNorm(hidden_states)
      │    │    │  hidden_states = MLP(hidden_states)
      │    │    │  hidden_states = RMSNorm(hidden_states)
      │    │    │  hidden_states = residual + hidden_states
      │    │    │
      │    │    └→ hidden_states [batch, seq_len, 2048]
      │    │
      │    └→ hidden_states = RMSNorm(hidden_states)   ← 步末 norm
      │
      │    hidden_states_list.append(hidden_states)
      │    gate_list.append(early_exit_gate(hidden_states))
      │       early_exit_gate = nn.Linear(2048, 1) → sigmoid → λ_t
      │
      └── ═════════════════════════════════

  返回: (BaseModelOutputWithPast, hidden_states_list[4], gate_list[4])
```

**五个关键设计细节**：

1. **Position embedding 只计算一次**，所有循环步共享。循环不引入额外的位置信息——它是一种"无时间维度"的深度迭代
2. **每步循环结束后**都对 hidden_states 做 RMSNorm，确保每步输出处于稳定数值范围
3. **KV cache 按二维索引展平为一维**：`cache_idx = current_ut × num_hidden_layers + layer_idx`。每个 UT 步的 KV 独立缓存
4. **循环之间没有 halting mechanism** — 所有 4 步都会完整执行。Exit gate 概率仅在 `OuroForCausalLM.forward` 中用于决定使用哪一步的输出
5. **返回值是三元组**（非标准单一输出）：`(BaseModelOutputWithPast, hidden_states_list, gate_list)`

### 7.4 Exit Gate 概率分布计算

门控网络极简：`nn.Linear(hidden_size, 1)` → sigmoid。

在 `OuroForCausalLM.forward` 中转化为完整概率分布：

```python
remaining_prob = 1.0
for i, gate in enumerate(gate_list):       # 4 个步
    lambda_i = sigmoid(gate).squeeze(-1)    # 该步退出率
    if i < last_step:
        p_i = lambda_i * remaining_prob     # 该步退出概率
        remaining_prob *= (1 - lambda_i)    # 剩余继续概率
    else:
        p_i = remaining_prob                # 最后一步必须全部退出
# sum(p_0, p_1, p_2, p_3) = 1.0
```

### 7.5 OuroForCausalLM.forward — 多策略推理

根据场景（训练 vs 推理）和配置采取不同策略：

**策略 1: 训练模式**（labels is not None）
```python
expected_logits = sum(p_i * lm_head(hidden_i) for i in 0..3)
loss = cross_entropy(expected_logits, labels)
```
对退出分布的期望进行优化，等价于 entropy-regularized objective。

**策略 2: exit_at_step（固定步退出）**
```python
logits = lm_head(hidden_states_list[exit_at_step])
```

**策略 3: exit_threshold（动态阈值退出）— 核心策略**
```python
cumulative_probs = cumsum(stacked_exit_pdf, dim=2)
threshold_mask = cumulative_probs >= threshold
exit_steps = argmax(threshold_mask.float(), dim=2)
# per-token 选择对应的 hidden state
stacked_hidden = stack(hidden_states_list, dim=2)
final_hidden = gather(stacked_hidden, exit_steps)
logits = lm_head(final_hidden)
```
每个 token 独立决定在哪一步退出——真正的自适应计算。

**策略 4: use_weighted_exit（加权平均）**
```python
expected_logits = sum(p_i * lm_head(hidden_i) for all steps)
```

**策略 5: 默认**
```python
logits = lm_head(last_hidden_state)
```

### 7.6 UniversalTransformerCache — 循环 KV 缓存

索引方案（1.4B 模型为例）：

```
UT Step 0: Layer  0 → cache[ 0], Layer  1 → cache[ 1], ... Layer 23 → cache[23]
UT Step 1: Layer  0 → cache[24], Layer  1 → cache[25], ... Layer 23 → cache[47]
UT Step 2: Layer  0 → cache[48], Layer  1 → cache[49], ... Layer 23 → cache[71]
UT Step 3: Layer  0 → cache[72], Layer  1 → cache[73], ... Layer 23 → cache[95]
```

1.4B 总共 96 个缓存槽位，2.6B 总共 192 个。

核心方法：
- `update(key, value, layer_idx)`：按需扩展缓存列表，拼接序列维度
- `get_seq_length(layer_idx)`：返回已缓存序列长度
- `reorder_cache(beam_idx)`：支持 beam search
- `clear()`：完全清空缓存

### 7.7 生成流程

1. HuggingFace 的 `generate()` 每步调用 `forward()`
2. `forward()` 内部执行完整 4 步 UT 循环
3. KV cache 在 `generate()` 步骤间通过 `past_key_values` 传递
4. 每个生成 token 内部，4 个 UT 步各自维护独立 KV

启用 Early Exit 推理：

```python
# 方式 1: 修改 config
model.config.early_exit_threshold = 0.8

# 方式 2: 通过 generate_kwargs
outputs = model.generate(input_ids, exit_threshold=0.8)

# 方式 3: 固定在某步退出
outputs = model.generate(input_ids, exit_at_step=2)
```

### 7.8 Tokenizer 分析

| 配置 | 值 |
|---|---|
| 类型 | GPT2Tokenizer (BPE) |
| 词表大小 | 49,152 |
| BOS/EOS token | "" (id=0) |
| 对话标记 | `<\|im_start\|>` (id=1), `<\|im_end\|>` (id=2) |
| model_max_length | 131,072 (128K) |
| Chat Template | ChatML 格式 |

### 7.9 与标准 HuggingFace 模型的差异总结

| 特性 | 标准 LLaMA | Ouro |
|---|---|---|
| 层执行方式 | 每层执行 1 次 | 每层循环执行 N 次（权重共享） |
| 归一化 | Pre-Norm（2 个/层） | Sandwich Norm（4 个/层） |
| KV Cache 索引 | 按 layer_idx | 按 (ut_step × num_layers + layer_idx) 展平 |
| 推理深度 | 固定 | 可通过 Exit Gate 动态 per-token 调整 |
| 最终输出 | 单一隐藏状态 | 多步隐藏状态列表 + gate 概率列表 |
| 损失计算 | 单点 cross_entropy | 多步加权期望 cross_entropy |
| Position Embedding | 每层各自计算 | 所有 UT 步共享同一组 cos/sin |
| forward 返回值 | BaseModelOutputWithPast | 三元组：(BaseModelOutputWithPast, hidden_states_list, gate_list) |
| RoPE 基频 | 10,000 | 1,000,000 |
| 代码骨架来源 | — | Qwen |

### 7.10 设计亮点与工程观察

1. **"全执行"设计**：所有 4 步都会完整执行，exit gate 仅在后续决定使用哪一步的输出。无法通过 early exit 节省前向计算量——节省的是 logits 计算和后续生成步骤，而非循环本身。若要真正节省计算，需在循环内部加入条件终止逻辑
2. **默认不启用 Early Exit**：`early_exit_threshold = 1.0` 意味着默认完全执行所有步，用户需主动配置
3. **极大的 RoPE 基频**：theta = 1,000,000（标准 LLaMA 为 10,000），配合 65K 最大位置编码，面向长上下文场景
4. **参数效率**：1.4B 参数通过 4 步循环等效 96 层网络，同等参数标准模型只能有约 24-32 层

---

## 8. 实验结果

### 8.1 评估配置

#### Base Model 评估设置

| 基准 | 设置 | 框架 |
|---|---|---|
| MMLU | logprobs, 5-shot | lm-eval-harness |
| MMLU-Pro | strict match, 5-shot CoT | lm-eval-harness |
| BBH | strict match, 3-shot CoT | lm-eval-harness |
| ARC-C | logprobs, 25-shot | lm-eval-harness |
| HellaSwag | logprobs, 10-shot | lm-eval-harness |
| Winogrande | logprobs, 5-shot | lm-eval-harness |
| GSM8K | strict match, 3-shot CoT | lm-eval-harness |
| MATH500 | strict match, 5-shot CoT | In-house |
| HumanEval / HumanEval+ | pass@1 | evalplus |
| MBPP / MBPP+ | pass@1 | evalplus |

#### Reasoning Model 评估设置

| 基准 | 协议 | 解码设置 |
|---|---|---|
| AIME 2024/2025 | In-house; LLM-as-judge | temp=1.0, top_p=0.7 |
| OlympiadBench | In-house; LLM-as-judge | temp=1.0, top_p=0.7 |
| GPQA | In-house; LLM-as-judge | temp=1.0, top_p=0.7 |
| SuperGPQA | In-house; LLM-as-judge | temp=1.0, top_p=0.7 |
| BeyondAIME | In-house; LLM-as-judge | temp=1.0, top_p=0.7 |
| HLE | In-house; LLM-as-judge | temp=1.0, top_p=0.7 |

### 8.2 基础模型评估

#### Ouro-1.4B vs 1-4B Baselines

| 基准 | Ouro-1.4B R4 | Qwen3-4B | Qwen2.5-3B | Llama3.2-3B |
|---|---|---|---|---|
| MMLU | **67.35** | 73.19 | 65.62 | 59.69 |
| MMLU-Pro | **48.62** | 51.40 | 37.87 | 33.34 |
| BBH | **71.02** | 70.95 | 55.37 | 39.45 |
| ARC-C | **60.92** | 63.65 | 55.46 | 52.47 |
| HellaSwag | **74.29** | 75.66 | 74.54 | 73.09 |
| Winogrande | **72.30** | 71.19 | 70.17 | 69.14 |
| GSM8K | **78.92** | 72.86 | 74.60 | 67.20 |
| MATH500 | **82.40** | 59.60 | 42.60 | 40.80 |
| HumanEval | **74.40** | 77.40 | 68.90 | 29.90 |
| HumanEval+ | **67.40** | 70.70 | 62.20 | 26.20 |
| MBPP | **73.00** | 78.80 | 63.00 | 50.30 |
| MBPP+ | **62.70** | 65.90 | 54.20 | 39.70 |

**结论**：1.4B Ouro 在推理任务上达到甚至超过 4B 模型水平（BBH 超越、GSM8K +6、MATH500 +22.8）。

#### Ouro-2.6B vs 3-12B Baselines

| 基准 | Ouro-2.6B R4 | Qwen3-8B | Qwen2.5-7B | Gemma3-12B |
|---|---|---|---|---|
| MMLU | **74.60** | 76.63 | 74.20 | 72.14 |
| MMLU-Pro | **55.73** | 53.72 | 43.55 | 49.21 |
| BBH | **80.46** | 77.65 | 53.72 | 78.41 |
| ARC-C | **66.40** | 66.10 | 63.65 | 72.44 |
| HellaSwag | **79.69** | 79.60 | 79.98 | 83.68 |
| Winogrande | **75.85** | 76.80 | 76.48 | 77.74 |
| GSM8K | **81.58** | 83.09 | 81.50 | 77.18 |
| MATH500 | **90.85** | 62.30 | 61.20 | 83.20 |
| HumanEval | **78.70** | 84.80 | 79.30 | 46.30 |
| HumanEval+ | **70.70** | 75.30 | 70.60 | 37.20 |
| MBPP | **80.40** | 79.00 | 73.80 | 73.50 |
| MBPP+ | **66.60** | 67.90 | 63.50 | 66.10 |

**结论**：2.6B Ouro 在多个推理密集基准上超越 8B 模型，MMLU-Pro/BBH/MATH500 大幅领先。

### 8.3 推理模型评估（Ouro-Thinking）

| 模型 | AIME24 p@1 | AIME24 p@10 | AIME25 p@1 | AIME25 p@10 | Olympiad | BeyondAIME | HLE | SuperGPQA | GPQA |
|---|---|---|---|---|---|---|---|---|---|
| Ouro-1.4B-Thinking R4 | 65.0 | 83.3 | 46.3 | 73.3 | 71.6 | 34.0 | 5.21 | 47.4 | 45.5 |
| Ouro-2.6B-Thinking R4 | 64.7 | 90.0 | 50.3 | 76.7 | 76.4 | 39.0 | 5.58 | 53.7 | 52.7 |
| Qwen3-4B | 61.3 | 75.0 | 51.3 | 63.3 | 73.2 | 31.0 | 5.21 | 51.9 | 54.5 |
| Qwen3-8B | 73.0 | 86.7 | 66.7 | 81.3 | 75.3 | 38.0 | 2.22 | 48.0 | 59.1 |
| Deepseek-Distill-1.5B | 29.6 | 66.7 | 23.0 | 43.3 | 56.4 | 9.0 | 4.20 | 26.5 | 33.2 |
| Deepseek-Distill-7B | 57.3 | 83.3 | 36.0 | 73.3 | 72.0 | 30.0 | 5.14 | 46.6 | 51.0 |

### 8.4 循环步数与外推性能

训练最大步数 T=4，测试 T=1 到 T=8：

#### Ouro-1.4B Base Model

| UT Step | ARC-C | MMLU | GSM8K | HellaSwag |
|---|---|---|---|---|
| 1 | 37.63 | 41.21 | — | 55.24 |
| 2 | 54.86 | 60.43 | — | 71.15 |
| 3 | 59.47 | 66.71 | — | 74.07 |
| **4（训练深度）** | **60.92** | **67.45** | **78.92** | **74.29** |
| 5（外推） | 58.96 | 66.64 | — | 73.72 |
| 8（外推） | 58.19 | 64.49 | — | 71.60 |

#### Ouro-1.4B-Thinking Model

| 基准 | T=1 | T=2 | T=3 | T=4 | T=5 | T=8 |
|---|---|---|---|---|---|---|
| OlympiadBench | 2.22 | 59.70 | 70.67 | 71.55 | 72.30 | 66.81 |
| AIME 2024 | 0.00 | 37.33 | 62.33 | 65.00 | 60.67 | 38.67 |
| AIME 2025 | 0.33 | 25.00 | 43.33 | 46.30 | 47.00 | 38.00 |

#### Ouro-2.6B-Thinking Model

| 基准 | T=1 | T=2 | T=3 | T=4 | T=5 | T=8 |
|---|---|---|---|---|---|---|
| OlympiadBench | 18.96 | 68.59 | 75.56 | 76.44 | 71.85 | 39.26 |
| AIME 2024 | 3.00 | 52.00 | 70.33 | 64.70 | 57.00 | 39.00 |
| AIME 2025 | 2.00 | 40.67 | 50.67 | 50.30 | 49.33 | 24.33 |

**关键发现**：
- T=1→2 有最显著的跳跃
- Reasoning 模型 T=1 性能极低，确认迭代对复杂推理至关重要
- 1.4B Thinking 模型在某些基准上 T=5 略优于 T=4
- 外推到 T=6-8 性能退化

### 8.5 自适应计算效率

三种提前退出策略比较（MMLU 上）：

1. **静态退出**（固定步数）— 基线
2. **隐藏状态差异阈值** — 监控 ∥h_t - h_{t-1}∥₂，性能具竞争力（差距 1-2%）
3. **学习式 Gate + Q-exit** — 最优方案

关键发现：
- 专项训练的 Gate 在所有计算预算下都取得最佳精度
- 在平均退出轮数 2.5 时，专项训练 Gate 达到 66% vs 标准 Gate 64%
- 1→2 轮（40%→60%）的跳跃远大于 3→4 轮（67.35%），说明自适应方法有效：大部分样本在中等深度即可达到近最优性能

### 8.6 KV Cache 共享

| 策略 | GSM8K | MATH-500 | 内存缩减 |
|---|---|---|---|
| Full (4× cache) | 78.92 | 82.40 | 1.00× |
| First-step only | 18.73 | 8.43 | 4.00× |
| **Last-step only** | **78.85** | **80.40** | **4.00×** |
| Averaged | 78.73 | 78.52 | 4.00× |

关键发现：推理阶段（decoding phase）可仅保留最后一轮 KV cache（GSM8K 仅差 0.07），内存减少 4 倍。第一轮 cache 不可用（性能崩溃）。Prefilling 阶段所有 4 步的 cache 都必须保留。

---

## 9. 机制理解：为什么 LoopLM 更优？

### 9.1 循环不增加知识容量

使用 Physics of LMs 框架的 Capo 任务（合成传记记忆）：
- LoopLM 和非循环模型的知识容量均约 **2 bits/parameter**
- 循环不增加参数的知识存储能力

实验设置：GPT-2 style 模型（1M-40M 参数），bioS(N) 数据集（N=20K-500K），1000 次曝光训练。

### 9.2 循环增强知识操纵能力

#### Mano 任务（模块化算术，F₂₃ 域）

| 模型 | L=10 | L=16 | L=24 |
|---|---|---|---|
| Base (12⊗1) | 93.6 | 94.4 | 34.8 |
| Base (2⊗1) | 21.5 | 8.4 | 7.5 |
| **Loop (2⊗6)** | **98.1** | **96.3** | **78.0** |
| Base (3⊗1) | 75.4 | 29.8 | 11.0 |
| **Loop (3⊗4)** | **97.9** | **95.8** | **92.2** |
| Base (6⊗1) | 84.7 | 59.5 | 20.0 |
| **Loop (6⊗2)** | **93.4** | **88.5** | **35.1** |

同参数量下，LoopLM 在知识操纵任务上大幅超越非循环模型。甚至 iso-FLOP 下也经常更优。

#### Multi-hop QA 任务

- 500 个实体名，20 个关系名，3-hop 问题
- 约 8×10⁵ 个可能的问题，3000 个测试问题
- LoopLM 在 3-hop 推理上用更少样本即可学会
- 更多循环 = 更快学习 + 更好性能

### 9.3 理论解释

**定理 1（非正式）**：给定知识图谱的邻接矩阵和查询对 (s,t)，存在一个独立于上下文图的 1 层 Transformer，循环 O(log₂D) 次即可检查从 s 到 t 是否存在路径（D 为图直径）。

| 潜在推理方法 | 顺序计算步骤 |
|---|---|
| Discrete CoT | O(n²) |
| Continuous CoT | O(D) |
| **Universal Transformer (LoopLM)** | **O(log D)** |

LoopLM 将顺序计算步从 O(n²) 指数级降低到 O(log D)，是最高效的潜在推理范式。

### 9.4 MMLU 子类别验证

- 最显著提升：初等数学 (+155.6%)、形式逻辑 (+143.3%)、逻辑谬误 (+127.8%)、高中统计 (+126.9%)
- 最小提升：道德场景 (+7.8%)、全球事实 (+8.3%)、病毒学 (+13.7%)、解剖学 (+21.4%)
- 证实：循环增强的是推理/操纵能力，而非知识检索

---

## 10. 安全性与忠实性

### 10.1 安全性（HEx-PHI 评估）

- 数据集：330 个例子，11 个禁止类别，GPT-4o 作为 judge
- Base 模型：greedy decoding, max_new_tokens=128
- Thinking 模型：temp=1.0, top_p=0.7, max_new_tokens=8192

结果：
- 循环步数增加 → 安全性提升（即使外推到 T>4 也成立）
- Ouro-1.4B-Thinking 有害率：**0.009**（与 Qwen3-4B-Thinking 持平）
- Ouro-2.6B-Thinking 有害率：**0.003**
- PCA 分析：更多循环步 → 更好区分良性/有害提示；不安全响应的点聚集在良性/有害集群边界附近

### 10.2 忠实性（Faithfulness）

- CoT 的已知问题：模型可能先决定答案，再用 CoT 合理化
- LoopLM 的推理基底是潜在状态序列 h^(1)→h^(2)→...→h^(T)
- 每次转换使用相同的共享权重块执行实质性计算

实验（Quora Question Pairs）：
- Step 2 vs Step 4 一致率仅 **36.1%**（说明中间推理确实在改变决策）
- 相邻步骤 i ≥ 4 时一致率接近 100%
- 对比 Qwen3-4B-Thinking 的 ROC AUC 为 0.99（思考过程几乎不影响结果），说明 Ouro 的潜在推理更真实地反映了决策过程

### 10.3 部署优势

1. **内置 draft model 用于投机解码**：浅层循环步 (Text(R_s)) 作为提议模型，深层 (Text(R_T)) 作为验证模型
2. **提前退出用于延迟控制**：简单输入快速退出，复杂输入分配更多计算
3. **监控与安全控制**：中间预测可用于预判输出安全性

---

## 11. Scaling Law 分析

### Total Loss Scaling Law

```
L_t = E + A/(N+t₁)^α + B/(D+t₂)^β + C/(Tm+t₃)^γ
```

- 变量：模型大小 N，训练数据 D，最大循环步 T_m
- 全数据点拟合 R² = **0.9596**
- 模型大小可泛化性：10 组合平均 R² = **0.9542**
- 训练数据可泛化性（25%/50%/75%）：R² = 0.9385 / 0.9609 / 0.962
- 最大循环步可泛化性：3 组平均 R² = **0.9581**

### Step-wise Loss Scaling Law

```
L_s = E + A/(N+t₁)^α + B/(D+t₂)^β + C/(T+t₃)^γ
```

- Tmax=2: R² = 0.8898
- Tmax=4: R² = 0.8146
- Tmax=8: R² = 0.795
- 拟合参数 γ 为正，确认循环步增加时损失递减

### Ponder 权重分布（Tmax=4, MMLU 推理时）

| 步骤 | 权重 |
|---|---|
| Step 1 | 0.0004 |
| Step 2 | 0.0855 |
| Step 3 | 0.3793 |
| Step 4 | 0.5348 |

模型将大部分权重分配给最深的步骤。

### LoopLM vs Standard Model 差距随模型大小变化

| 模型大小 | Step 2 差距 | Step 4 差距 |
|---|---|---|
| 170M | 0.021 | 0.039 |
| 340M | 0.023 | 0.037 |
| 680M | 0.015 | 0.026 |
| 1.3B | 0.017 | 0.025 |

差距定义为 (Standard score - LoopLM score)，随模型增大而缩小，说明 LoopLM 在更大模型上更接近 Standard 模型。

---

## 12. 使用指南

### 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ByteDance/Ouro-1.4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 调整循环步数

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("ByteDance/Ouro-1.4B")
config.total_ut_steps = 3  # 使用 3 个循环步
model = AutoModelForCausalLM.from_pretrained(
    "ByteDance/Ouro-1.4B",
    config=config,
    device_map="auto"
)
```

### 配置项

- `total_ut_steps`：控制循环步数（默认 4），可调节性能/计算时间权衡
- `early_exit_threshold`：控制自适应退出（默认 1.0），更低值鼓励更早退出

### 兼容性

- **必须使用 transformers < 4.56.0**（推荐 4.54.1）
- vLLM/SGLang 集成已就绪（2025-10-30 公告）
- vLLM 目前不支持自适应提前退出

### HuggingFace 模型统计

| 模型 | 下载量（月） | 总下载量 | Likes |
|---|---|---|---|
| ByteDance/Ouro-1.4B | 57,188 | 161,029 | 92 |
| ByteDance/Ouro-2.6B | 5,750 | 48,786 | 82 |

---

## 13. 核心贡献总结

1. **规模化参数效率**：1.4B/2.6B LoopLM 匹配 4B/8B 标准 Transformer（2-3× 参数效率提升）
2. **熵正则化自适应计算**：均匀先验 + 专项训练实现动态深度分配
3. **机制理解**：循环不增加知识存储（~2 bits/param），但显著增强知识操纵能力；理论证明 LoopLM 可在 O(log D) 步完成知识图谱搜索
4. **安全性与忠实性**：更多循环步提升安全性；潜在推理比显式 CoT 更忠实
5. **将循环深度确立为第三缩放轴**：超越模型大小和数据量

---

## 14. 局限与未来方向

1. **RL 对齐尚未成功**：vLLM/SGLang 的固定执行路径与动态深度不兼容，需开发新基础设施
2. **外推退化**：超过训练深度的循环步导致性能下降
3. **代码尚未开源**（截至 2025 年 11 月声明 Coming Soon）
4. **循环步稳定性**：8 步循环训练不稳定，需进一步研究
5. **Early Exit 不节省前向计算**：所有循环步都完整执行，gate 仅选择使用哪一步的输出
6. **理论理解待深化**：为何循环改善操纵但非存储？为何样本效率更高？

---

## 15. 引用

```bibtex
@article{zhu2025scaling,
  title={Scaling Latent Reasoning via Looped Language Models},
  author={Zhu, Rui-Jie and Wang, Zixuan and Hua, Kai and Zhang, Tianyu and Li, Ziniu and Que, Haoran and Wei, Boyi and Wen, Zixin and Yin, Fan and Xing, He and others},
  journal={arXiv preprint arXiv:2510.25741},
  year={2025}
}
```

---

*文档生成日期：2026-05-28*
