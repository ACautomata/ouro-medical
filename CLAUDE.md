# CLAUDE.md — mobius

## Project Overview

mobius 是一个基于 Ouro Looped Language Model 的医学领域研究项目。

- **核心依赖**：[Ouro](https://huggingface.co/ByteDance/Ouro-1.4B) — ByteDance Seed 出品的循环语言模型（arXiv:2510.25741）
- **Ouro 代码位置**：`src/mobius/ouro/`（vendored，从 HuggingFace 提取）
- **论文资料**：`docs/Ouro论文完整信息汇总.md`
- **依赖**：PyTorch, Transformers, MONAI

## Architecture

Ouro 的核心创新是 **Looped Language Model (LoopLM)**：将同一组 Transformer 层循环执行多次（默认 4 次），在潜空间中进行迭代推理，配合 Exit Gate 实现自适应计算深度。

关键组件：
- `OuroModel` — 核心循环引擎，执行 UT 循环 + Exit Gate
- `OuroForCausalLM` — 语言模型头，支持多种推理策略（固定步/阈值退出/加权平均）
- `UniversalTransformerCache` — 二维索引 KV 缓存（ut_step × layer_idx）
- `OuroDecoderLayer` — Sandwich RMSNorm（4 个 norm/层）
- `OuroConfig` — 含 `total_ut_steps`、`early_exit_threshold` 等 Ouro 特有参数

## Setup

```bash
pip install -e ".[dev]"
```

注意：需要 `transformers>=4.55.0,<4.56.0`。

## Vendored Code Copyright

`src/mobius/ouro/` 下的代码源自：
- Qwen team / Alibaba Group（configuration 骨架）— Apache-2.0
- ByteDance Seed（UT 循环机制、Exit Gate、UniversalTransformerCache）— Apache-2.0

修改时保留文件头部的版权声明。
