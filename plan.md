# nanoGPT 项目运行与拓展指南

## 一、项目概述

nanoGPT 是 Andrej Karpathy 写的一个最简 GPT 训练/微调代码库，核心文件仅两个：
- `model.py` (~330 行): GPT 模型定义（GPTConfig, GPT, CausalSelfAttention, MLP, Block, LayerNorm）
- `train.py` (~330 行): 训练循环（数据加载、优化器、DDP、checkpoint、采样等）

支持的三种模式：
1. **从零训练** (scratch): `init_from = 'scratch'`
2. **从 checkpoint 恢复训练** (resume): `init_from = 'resume'`
3. **微调/评估 GPT-2 预训练模型**: `init_from = 'gpt2'` / `'gpt2-medium'` / `'gpt2-large'` / `'gpt2-xl'`

---

## 二、运行步骤

### 2.1 环境安装

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖说明：
| 包 | 用途 |
|---|---|
| `torch` (>=2.0 推荐) | 深度学习框架，需要 `torch.compile` 和 Flash Attention |
| `numpy` | 数据文件读写（memmap） |
| `transformers` | 加载 HuggingFace GPT-2 预训练权重 |
| `datasets` | 下载和预处理 OpenWebText 数据集 |
| `tiktoken` | OpenAI GPT-2 BPE 分词器 |
| `wandb` | 可选，训练日志记录 |
| `tqdm` | 进度条 |

### 2.2 快速体验（Character-level Shakespeare，适合 CPU/单 GPU）

**Step 1: 准备数据**
```bash
python data/shakespeare_char/prepare.py
```
这会下载莎士比亚文本（~1MB），将字符映射为整数，生成 `data/shakespeare_char/train.bin`、`val.bin` 和 `meta.pkl`（vocab_size = 65）。

**Step 2: 训练**

有 GPU:
```bash
python train.py config/train_shakespeare_char.py
```
3 分钟训练，loss 约 1.47，模型配置：6 层、6 头、384 维、block_size=256。

仅 CPU / Macbook:
```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu --compile=False --eval_iters=20 --log_interval=1 \
    --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 \
    --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Apple Silicon Mac:
```bash
python train.py config/train_shakespeare_char.py --device=mps
```

**Step 3: 推理采样**
```bash
python sample.py --out_dir=out-shakespeare-char
```
从训练好的模型生成文本。

---

### 2.3 微调 GPT-2 做 Shakespeare（BPE tokenizer）

**Step 1: 准备数据**
```bash
python data/shakespeare/prepare.py
```
使用 GPT-2 BPE tokenizer 编码，生成 `data/shakespeare/train.bin`、`val.bin`。

**Step 2: 微调**
```bash
python train.py config/finetune_shakespeare.py
```
从 `gpt2-xl` 初始化，在 Shakespeare 上微调约 20 步（learning_rate=3e-5，constant LR）。

**Step 3: 采样**
```bash
python sample.py --out_dir=out-shakespeare
```

---

### 2.4 复现 GPT-2 124M（OpenWebText，需要 8×A100 40GB）

**Step 1: 准备数据**
```bash
python data/openwebtext/prepare.py
```
下载 OpenWebText（~54GB cache），GPT-2 BPE 分词后生成：
- `train.bin`: ~17GB, ~9B tokens
- `val.bin`: ~8.5MB, ~4.4M tokens

**Step 2: 训练**
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```
约 4 天训练，loss 降至 ~2.85。使用 DDP（分布式数据并行），总 batch size = 12 × 1024 × 5 × 8 = 491,520 tokens/iter。

多节点训练：
```bash
# 主节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=123.456.123.456 --master_port=1234 train.py
# 工作节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=123.456.123.456 --master_port=1234 train.py
```
若没有 InfiniBand 网络，需要 `NCCL_IB_DISABLE=1`。

**Step 3: 评估 GPT-2 baseline**
```bash
python train.py config/eval_gpt2.py         # 124M
python train.py config/eval_gpt2_medium.py  # 350M
python train.py config/eval_gpt2_large.py   # 774M
python train.py config/eval_gpt2_xl.py      # 1558M
```

---

## 三、核心代码架构

```
nanoGPT/
├── model.py                    # GPT 模型定义
│   ├── LayerNorm               # 支持可选 bias 的 LayerNorm
│   ├── CausalSelfAttention     # 多头自注意力（Flash Attention + 手动实现双路径）
│   ├── MLP                     # FeedForward（4x 扩展 + GELU）
│   ├── Block                   # Transformer Block = LN + Attn + LN + MLP（Pre-LN）
│   ├── GPTConfig               # @dataclass: block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias
│   └── GPT                     # 主模型
│       ├── __init__            # wte + wpe + blocks + ln_f + lm_head（weight tying）
│       ├── forward()           # 前向传播，可选计算 loss
│       ├── from_pretrained()   # 加载 HuggingFace GPT-2 权重
│       ├── configure_optimizers()  # AdamW + 分组 weight decay
│       ├── estimate_mfu()      # 估算 MFU（Model FLOPs Utilization）
│       ├── generate()          # 自回归文本生成（temperature + top-k）
│       └── crop_block_size()   # 模型手术：减小 block_size
├── train.py                    # 训练脚本
│   ├── 配置参数（~40个，可通过 config 文件或命令行覆盖）
│   ├── DDP 初始化
│   ├── get_batch()             # 从 memmap 文件随机采样 batch
│   ├── estimate_loss()         # 评估 train/val loss
│   ├── get_lr()                # Cosine 学习率衰减 + linear warmup
│   └── 训练循环                  # forward + backward + gradient_accumulation + GradScaler
├── sample.py                   # 推理采样脚本
├── bench.py                    # 性能基准测试脚本
├── configurator.py             # 配置系统：通过文件或 --key=value 覆盖全局变量
├── config/                     # 配置文件目录
│   ├── train_shakespeare_char.py    # 字符级 Shakespeare 训练
│   ├── train_gpt2.py                # GPT-2 124M 复现训练
│   ├── finetune_shakespeare.py      # GPT-2 → Shakespeare 微调
│   ├── eval_gpt2.py                 # GPT-2 评估
│   ├── eval_gpt2_medium.py
│   ├── eval_gpt2_large.py
│   └── eval_gpt2_xl.py
└── data/                       # 数据集目录
    ├── shakespeare_char/       # 字符级 Shakespeare
    ├── shakespeare/            # BPE 级 Shakespeare
    └── openwebtext/            # OpenWebText（GPT-2 复现用）
```

---

## 四、关键设计要点

### 4.1 配置系统 (`configurator.py`)
- 所有可配置参数都是 `train.py` 中的**全局变量**，有默认值
- 通过 `python train.py config/xxx.py --key=value` 覆盖
- 配置文件的执行顺序：先执行 config 文件，再用命令行 `--key=value` 覆盖
- 命令行参数值需与默认值类型一致（用 `literal_eval` 解析）

### 4.2 数据格式
- 所有数据集最终被处理成 `.bin` 文件（`np.uint16` 的 token ID 序列，raw bytes）
- 通过 `np.memmap` 零拷贝读取，每次随机采样一个 `block_size` 长的连续序列
- `meta.pkl` 包含 `vocab_size`、`stoi`/`itos` 编解码映射

### 4.3 模型架构
- **GPT-2 架构**：Pre-LayerNorm Transformer
- **Weight Tying**: `wte.weight` 与 `lm_head.weight` 共享
- **Flash Attention**: 自动检测 `torch.nn.functional.scaled_dot_product_attention`，若不可用则退化为手动实现
- **GELU 激活**（非 ReLU）
- **无 bias 选项**：默认 `bias=False`（比 GPT-2 原始 True 更快更好）

### 4.4 训练优化
- **混合精度**: `bfloat16` 优先，回退 `float16`（带 GradScaler）
- **`torch.compile`**: PyTorch 2.0 编译加速（~2x）
- **梯度累积**: `gradient_accumulation_steps` 模拟更大的 batch size
- **DDP**: 多卡/多机分布式训练，只在最后一个 micro_step 同步梯度
- **Async 数据预取**: 在当前 forward 时异步加载下一个 batch

### 4.5 学习率调度
```
          warmup_iters          lr_decay_iters
lr:  0 ------------> learning_rate ------------> min_lr
      linear warmup     cosine decay              (lr / 10)
```

---

## 五、拓展方向与具体做法

### 5.1 使用自定义数据集

**步骤：**
1. 在 `data/` 下创建新目录，如 `data/my_dataset/`
2. 仿照 `data/shakespeare/prepare.py` 写一个 `prepare.py`：
   - 读取你的原始文本数据
   - 选择合适的 tokenizer（tiktoken BPE / 自定义字符映射 / 其他）
   - 按 90/10 划分 train/val
   - 编码为 int 序列，导出 `train.bin`、`val.bin`（`np.uint16`）
   - 如有自定义 tokenizer，保存 `meta.pkl`
3. 在 `config/` 下创建配置文件，修改 `dataset = 'my_dataset'`
4. 运行 `python train.py config/my_config.py`

### 5.2 修改模型架构

**在 `model.py` 中修改：**

| 方向 | 修改位置 | 说明 |
|---|---|---|
| 添加 RoPE / ALiBi 位置编码 | `GPT.forward()` 和 `CausalSelfAttention.forward()` | 替换 `wpe`，在 attention score 上加 bias |
| 改为 GQA / MQA | `CausalSelfAttention.__init__()` 和 `forward()` | 修改 QKV 的维度映射关系 |
| 添加 MoE 层 | `MLP` 或新建 `MoEBlock` | 将 FFN 替换为多个 expert + router |
| 添加 RMSNorm | 新建 `RMSNorm(nn.Module)` | 替换 `LayerNorm` |
| 改为 SwiGLU 激活 | 修改 `MLP.__init__()` 和 `forward()` | 3 个权重矩阵代替 2 个 |
| 支持 Flash Attention 2/3 | 升级 `scaled_dot_product_attention` 调用 | 可能需升级 PyTorch 版本 |
| 添加 KV Cache | `CausalSelfAttention` 和 `GPT.generate()` | 推理加速，需要维护 past_key_values |
| 增大模型规模 | 修改 `GPTConfig` 参数或配置文件的 `n_layer/n_head/n_embd` | 参考 `transformer_sizing.ipynb` |

### 5.3 修改训练策略

**在 `train.py` 中修改：**

| 方向 | 修改位置 | 说明 |
|---|---|---|
| 更换优化器 | `optimizer = model.configure_optimizers(...)` | 替换为 Sophia / Lion / SGD 等 |
| 调整 LR schedule | `get_lr()` 函数 | Cosine → Linear / Polynomial / OneCycle |
| 添加 Gradient Clipping 策略 | grad_clip 附近 | 自适应 gradient clipping |
| 批次大小线性增长 | 训练循环开始时 | 逐渐增加 `batch_size` 或 `block_size` |
| 知识蒸馏 | `forward()` 后 | 加 KL divergence loss 与 teacher model |
| RLHF / DPO 微调 | 新增训练脚本 | 需要额外的 reward model / preference data |

### 5.4 添加新的采样/推理功能

**在 `sample.py` 或 `model.py` 的 `generate()` 中修改：**

| 方向 | 说明 |
|---|---|
| Beam Search | 替换贪心/采样解码 |
| Top-p (nucleus) sampling | 在 `generate()` 中添加 `top_p` 参数 |
| Classifier-Free Guidance | 需要条件输入 + guidance scale |
| Speculative Decoding | 需要 draft model + 验证循环 |
| Chat 格式对话 | 包装 system/user/assistant 模板 |
| Batch generation | 修改 `generate()` 支持不同长度的 batch |

### 5.5 添加新的评估指标

**在 `train.py` 的 `estimate_loss()` 附近添加：**

```python
@torch.no_grad()
def evaluate_perplexity():
    # 计算 perplexity = exp(loss)
    pass

@torch.no_grad()
def evaluate_hellaswag():
    # HellaSwag 常识推理评估
    pass
```

### 5.6 性能优化

| 方向 | 说明 |
|---|---|
| FSDP (替代 DDP) | `torch.distributed.fsdp` 支持更大模型分片 |
| Activation Checkpointing | `torch.utils.checkpoint` 减少显存 |
| Mixed Precision 微调 | 调整 `ptdtype` 和 `GradScaler` 配置 |
| DataLoader 改进 | 使用 `torch.utils.data.DataLoader` 替代 memmap 随机采样 |
| DeepSpeed / Megatron 集成 | 引入 ZeRO 优化等 |

### 5.7 添加 LoRA / QLoRA 微调

1. 安装 `peft` 库
2. 在 `model.py` 的 `GPT.__init__()` 后注入 LoRA adapter
3. 冻结原始权重，只训练 adapter
4. 大幅降低显存需求，适合在消费级 GPU 上微调大模型

### 5.8 添加 WandB / TensorBoard 之外的回调

在 `train.py` 的 logging 部分添加自定义回调：
- 定期生成样本并记录
- 保存 attention 可视化
- 梯度直方图
- 自定义 early stopping 策略

---

## 六、关键注意事项

1. **PyTorch 版本**: 建议 >= 2.0，以获得 `torch.compile` 和 Flash Attention 加速
2. **数据格式**: `.bin` 文件是 raw uint16 序列，不是标准格式；确保 prepare 脚本和训练脚本的 dtype 一致
3. **vocab_size**: 从 scratch 训练时，默认 50304（= 50257 向上取整到 64 的倍数，优化 CUDA kernel 效率）
4. **block_size**: 可通过命令行 `--block_size` 覆盖配置文件的值；如果加载的 checkpoint 有更大的 block_size，会自动 crop
5. **DDP 下的梯度累积**: `gradient_accumulation_steps` 必须是 `world_size` 的整数倍
6. **已废弃**: 作者表示此项目已被 [nanochat](https://github.com/karpathy/nanochat) 替代，但 nanoGPT 仍是一个优秀的学习和实验起点
