# SenseNova-U1 / NEO-unify 推理代码参考

> 来源: [GitHub - OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)

## 一、环境安装

```bash
uv pip install -e ".[gguf]"
```

Docker 部署：
```bash
docker pull lightx2v/lightllm_lightx2v:20260407
```

## 二、视觉问答 (VQA)

```python
python examples/vqa/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --image examples/vqa/data/images/menu.jpg \
  --question "My friend and I are dining together tonight..." \
  --output outputs/answer.txt \
  --max_new_tokens 8192
```

## 三、文生图 (Text-to-Image)

```python
python examples/t2i/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "your prompt here" \
  --width 2720 --height 1536 \
  --cfg_scale 4.0 --num_steps 50
```

## 四、图文交错生成

```python
python examples/interleave/inference.py \
  --model_path sensenova/SenseNova-U1-8B-MoT \
  --prompt "I want to learn how to cook..." \
  --resolution "16:9"
```

## 五、显存优化方案

### 5.1 GGUF 量化

```bash
--gguf_checkpoint /path/to/model-Q4_K_M.gguf
```

### 5.2 VRAM 分层加载

```bash
--vram_mode balanced  # 推荐用于10-12GB消费级GPU
--vram_mode low       # 最低显存占用
```

**推荐配置**: Q4 GGUF + balanced 模式

## 六、推理参数说明

| 参数 | 说明 |
|------|------|
| `--model_path` | HuggingFace 模型路径或本地路径 |
| `--image` | 输入图像路径 (VQA模式) |
| `--question` | 问题文本 (VQA模式) |
| `--prompt` | 生成提示词 (T2I模式) |
| `--width/height` | 输出图像分辨率 |
| `--cfg_scale` | CFG 引导强度 |
| `--num_steps` | 推理步数 |
| `--max_new_tokens` | 最大生成长度 |
| `--gguf_checkpoint` | GGUF 量化模型路径 |
| `--vram_mode` | 显存优化模式 (balanced/low) |
| `--output` | 输出文件路径 |

## 七、已知限制

- 最大上下文长度仅 32K tokens
- 复杂交互场景下人体细节可能不完美
- 文本渲染偶有拼写/格式问题
