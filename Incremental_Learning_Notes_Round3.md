# 第三轮增量学习笔记

> 基于 8x4090 复现计划制定过程中新发现的知识点
> 对比前两轮文档的增量学习内容

---

## 目录

- [1. 资源规划与工程权衡](#1-资源规划与工程权衡)
- [2. 训练流程深层理解](#2-训练流程深层理解)
- [3. 分布式训练实践细节](#3-分布式训练实践细节)
- [4. 数据管线深入](#4-数据管线深入)
- [5. 评估与验证体系](#5-评估与验证体系)
- [6. 实际复现中的关键决策](#6-实际复现中的关键决策)

---

## 1. 资源规划与工程权衡

### 1.1 GPU 显存预算的精确计算

**新发现**：之前只了解模型参数量，现在理解了完整的显存预算

```
训练显存 = 模型参数 + 梯度 + 优化器状态 + 激活值 + 临时缓冲区

以 8B 模型为例（FP16）：
- 模型参数：8B × 2 bytes = 16GB
- 梯度：8B × 2 bytes = 16GB
- AdamW 优化器状态：8B × 8 bytes = 64GB（m + v + step）
- 激活值：取决于 batch size 和序列长度，约 10-30GB
- 总计：约 100-130GB

以 1.5B 模型为例（FP16）：
- 模型参数：1.5B × 2 bytes = 3GB
- 梯度：1.5B × 2 bytes = 3GB
- AdamW 优化器状态：1.5B × 8 bytes = 12GB
- 激活值：约 2-5GB
- 总计：约 20-23GB
```

**关键洞察**：
- 4090 的 24GB 显存刚好够训练 1.5B 模型
- 8B 模型必须使用 FSDP 分片或多卡并行
- 推理显存需求远小于训练（只需要模型参数 + KV Cache）

### 1.2 在线 vs 离线模式的权衡

**新发现**：之前只了解概念，现在理解了实际的资源分配策略

| 维度 | 在线模式 | 离线模式 |
|------|---------|---------|
| **显存峰值** | vLLM + 训练 = 双倍 | vLLM 或 训练 = 单倍 |
| **GPU 利用率** | 需要分组 GPU | 可以复用 GPU |
| **数据新鲜度** | 实时生成 | 预生成 |
| **调试难度** | 难（两个系统同时运行） | 易（分步执行） |
| **适合场景** | 大 GPU 集群 | 小 GPU 集群 |

**实际决策**：
```
8x4090 环境下：
- 优先使用离线模式（显存更可控）
- 在线模式需要至少 4 卡给 vLLM，4 卡给训练
- 离线模式可以先用 2 卡跑 vLLM，再用 4 卡训练
```

### 1.3 模型大小选择的考量

**新发现**：选择模型大小不仅是性能问题，更是资源约束问题

```
模型选择决策树：
1.5B → 完全可行，适合学习流程
3B   → 可行，效果与资源平衡
7B   → 需要优化（量化/梯度检查点）
13B+ → 需要多机或使用量化推理
```

**关键代码**：`scripts/train.py` 中的显存优化选项
```python
# 梯度检查点
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# 混合精度
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,  # 参数用 bf16
    reduce_dtype=torch.float32,  # 通信用 fp32
)
```

---

## 2. 训练流程深层理解

### 2.1 Multipack Batch Sampler 的设计

**新发现**：之前不了解的高效数据打包机制

```python
# 实现：src/speculators/train/distributed_batch_sampler.py
class MultipackDistributedBatchSamplerV2(Sampler):
    """
    使用 LPT（Longest Processing Time First）算法
    将不同长度的样本打包到固定容量的 batch 中
    """
```

**核心算法**：
```
问题：如何将不同长度的样本分配到多个 GPU 的 batch 中？
约束：每个 batch 的总 token 数不能超过 max_len

LPT 算法：
1. 按样本长度降序排列
2. 依次将每个样本分配到当前最空的 batch
3. 如果样本放不下，开新的 batch

时间复杂度：O(n log n + n log replicas)
```

**为什么重要**：
- 避免 padding 浪费（传统方法需要 padding 到相同长度）
- 提高 GPU 利用率（每个 batch 的 token 数接近满载）
- 支持动态序列长度

**代码位置**：`src/speculators/train/distributed_batch_sampler.py:47`
```python
def _lpt_packed_batch(lengths, max_len, num_replicas, start_index, rank):
    """使用 LPT 算法打包样本"""
    local_batch = []
    heap = [_Bin(0, i) for i in range(num_replicas)]
    indices = np.argsort(lengths)[::-1]  # 降序排列
    
    for idx, size in zip(indices, lengths[indices]):
        new_fill = heap[0].fill + size
        if new_fill > max_len:
            return None  # 放不下
        
        if heap[0].rank == rank:
            local_batch.append(start_index + idx)
        
        heapreplace(heap, _Bin(new_fill, heap[0].rank))
    
    return local_batch
```

### 2.2 Noise Transform 的作用

**新发现**：训练时添加噪声可以提高鲁棒性

```python
# 实现：src/speculators/train/noise_transforms.py
class AddGaussianNoise(TransformTensors):
    def transform(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

class AddUniformNoise(TransformTensors):
    def transform(self, tensor):
        return tensor + 2 * (torch.rand_like(tensor) - 0.5) * self.std
```

**为什么需要噪声**：
1. **防止过拟合**：Hidden states 来自固定的数据集，添加噪声增加多样性
2. **提高鲁棒性**：推理时的 hidden states 可能与训练数据有偏差
3. **正则化效果**：类似 Dropout，但作用在输入而非权重

**使用场景**：
```python
# 在线训练时，hidden states 来自 vLLM，可能有噪声
# 离线训练时，hidden states 是预生成的，更稳定
# 两种情况都可以添加噪声提高鲁棒性
```

### 2.3 Vocabulary Mapping 的设计

**新发现**：Draft vocabulary 可以小于 target vocabulary

```python
# 实现：src/speculators/train/vocab_mapping.py
def build_vocab_mappings_from_distribution(
    token_freq_dict, draft_vocab_size, target_vocab_size
):
    # 1. 按频率排序 token
    sorted_tokens = sorted(token_freq_dict, key=lambda tid: (-token_freq_dict[tid], tid))
    
    # 2. 选择最常见的 token
    selected_token_ids = sorted_tokens[:draft_vocab_size]
    
    # 3. 创建映射
    draft_to_target = torch.tensor(selected_token_ids) - torch.arange(draft_vocab_size)
    target_to_draft = torch.zeros(target_vocab_size, dtype=torch.bool)
    target_to_draft[selected_token_ids] = True
    
    return draft_to_target, target_to_draft
```

**为什么使用较小的 draft vocabulary**：
1. **减少计算量**：LM head 的计算量与 vocab size 成正比
2. **减少内存**：Embedding 层的参数量 = vocab_size × hidden_size
3. **提高效率**：大部分 token 可以用少量常见 token 覆盖

**实际效果**：
```
Target vocab size: 151936 (Qwen2.5)
Draft vocab size: 32000 (约 21%)
参数减少：~240M → ~48M（LM head）
```

---

## 3. 分布式训练实践细节

### 3.1 FSDP 的 Mixed Precision 策略

**新发现**：FSDP 支持细粒度的精度控制

```python
# 实现：src/speculators/train/utils.py
def apply_fully_sharded(model):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,   # 参数存储用 bf16
        reduce_dtype=torch.float32,   # 梯度聚合用 fp32
    )
    
    # 逐层包装
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    
    # 包装整个模型
    fully_shard(model)
```

**为什么这样设计**：
1. **参数用 bf16**：减少显存占用，提高计算速度
2. **梯度用 fp32**：保证梯度聚合的精度，避免累积误差
3. **逐层包装**：实现更细粒度的分片，减少通信开销

### 3.2 Checkpoint 的分布式保存与加载

**新发现**：分布式 checkpoint 需要特殊的处理

```python
# 实现：src/speculators/train/checkpointer.py
class DistributedCheckpointer(BaseCheckpointer):
    def save_checkpoint(self, model, optimizer, epoch, float_dtype=torch.bfloat16):
        # 1. 收集完整 state dict（从所有 rank）
        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        
        # 2. 只在 rank 0 保存
        if dist.get_rank() == 0:
            model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
            torch.save(optimizer_state_dict, self.optimizer_path(epoch))
        
        # 3. 同步所有 rank
        dist.barrier()
    
    def load_model_state_dict(self, model, float_dtype=None):
        # 1. 加载完整 state dict
        full_state_dict = load_safetensors_state_dict(...)
        
        # 2. 广播到所有 rank
        set_model_state_dict(
            model, full_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False
            )
        )
        dist.barrier()
```

**关键设计决策**：
- **CPU Offload**：在 GPU 上收集 state dict 会 OOM，需要 offload 到 CPU
- **Rank 0 保存**：避免多进程同时写文件导致冲突
- **Barrier 同步**：确保所有 rank 都完成后再继续

### 3.3 Graceful Shutdown 机制

**新发现**：训练中断时需要保存 checkpoint

```python
# 实现：src/speculators/train/graceful_shutdown.py
class GracefulShutdownHandler:
    def __init__(self, timeout=120):
        self._interrupted = False
        self._timeout = timeout
    
    def _handler(self, signum, frame):
        # 只在主进程处理信号
        if os.getpid() != self._owner_pid:
            return
        
        # 第一次中断：抛出异常，尝试保存 checkpoint
        if not self._interrupted:
            self._interrupted = True
            raise TrainingInterruptedError(signum)
        
        # 第二次中断：忽略（防止 torchrun 重复发送）
    
    @property
    def timeout(self):
        return self._timeout

def with_graceful_shutdown(save_label="interrupted", timeout=120):
    """装饰器：训练中断时自动保存 checkpoint"""
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            handler = GracefulShutdownHandler(timeout=timeout)
            handler.install()
            
            try:
                return fn(self, *args, **kwargs)
            except TrainingInterruptedError:
                handler.restore()
                
                # 启动看门狗定时器
                timer = threading.Timer(handler.timeout, _watchdog)
                timer.start()
                
                try:
                    self.maybe_save_checkpoint(save_label)
                finally:
                    timer.cancel()
        
        return wrapper
    return decorator
```

**为什么需要这个机制**：
1. **长时间训练**：训练可能持续数小时或数天
2. **资源抢占**：共享集群可能被抢占
3. **意外中断**：网络故障、硬件故障等
4. **用户中断**：Ctrl+C 手动停止

---

## 4. 数据管线深入

### 4.1 Hidden States 的存储格式

**新发现**：Hidden states 使用 safetensors 格式存储

```python
# 数据结构
{
    "hidden_states": [seq_len, num_layers, hidden_size],  # 多层 hidden states
    "token_ids": [seq_len],                               # 对应的 token IDs
}

# 文件格式：hs_{index}.safetensors
# 优点：安全性、内存映射、跨框架兼容
```

**显存占用估算**：
```
以 Qwen2.5-1.5B 为例：
- hidden_size = 1536
- num_layers = 3（选择的中间层）
- seq_len = 4096

单个样本：4096 × 3 × 1536 × 2 bytes = 37.7MB
5000 个样本：~184GB

以 Qwen2.5-3B 为例：
- hidden_size = 2048
- num_layers = 3
- seq_len = 8192

单个样本：8192 × 3 × 2048 × 2 bytes = 100.6MB
10000 个样本：~983GB
```

**磁盘空间规划**：
```
1.5B 模型 + 5k 样本：~200GB
3B 模型 + 10k 样本：~1TB
7B 模型 + 10k 样本：~2TB
```

### 4.2 Online Training 的数据流

**新发现**：在线训练的数据流比想象的复杂

```
数据流：
1. DataLoader 从 ArrowDataset 读取样本
2. 检查 hidden states 是否存在
3. 如果不存在，调用 vLLM 生成
4. vLLM 返回 hidden states 文件路径
5. 加载 hidden states 文件
6. 验证 token IDs 一致性
7. 应用 transform（添加噪声等）
8. 组装 batch
9. 送入模型训练
```

**关键代码**：`src/speculators/train/data.py:309`
```python
def _maybe_generate_hs(self, index):
    """如果 hidden states 不存在，调用 vLLM 生成"""
    if not self.client:
        self._setup_client()
    
    dataset_item = self.data[index]
    client_item = build_client_item(dataset_item)
    
    try:
        # 调用 vLLM 生成 hidden states
        hs_filepath = generate_hidden_states(
            self.client, self.model, client_item,
            timeout=self.request_timeout,
            max_retries=self.max_retries
        )
        
        # 加载并验证
        loaded_hs = _maybe_load_hs_file(Path(hs_filepath))
        check_hidden_states(loaded_hs, dataset_item["input_ids"].tolist())
        
        # 根据配置决定是否缓存
        match self.on_generate:
            case "cache":
                shutil.move(hs_filepath, target_path)
            case "delete":
                Path(hs_filepath).unlink()
        
        return loaded_hs
    except Exception as e:
        warnings.warn(f"Failed to generate hidden states: {e}")
        return None
```

### 4.3 Token Frequency Distribution 的用途

**新发现**：Token 频率用于构建 draft vocabulary

```python
# 实现：src/speculators/train/vocab_mapping.py
def save_token_frequency_distribution(dataset, output_path):
    """统计训练数据中 token 的频率"""
    token_freq = Counter()
    for item in dataset:
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        # 只统计 assistant token（loss_mask=1 的位置）
        masked_token_ids = input_ids[loss_mask.to(torch.bool)]
        unique_ids, counts = masked_token_ids.unique(return_counts=True)
        token_freq.update(dict(zip(unique_ids.tolist(), counts.tolist())))
    
    torch.save(dict(token_freq), output_path)
```

**为什么只统计 assistant token**：
1. **训练目标**：Draft model 只预测 assistant 的回复
2. **效率**：User 的 token 不需要预测
3. **质量**：关注生成质量而非理解质量

---

## 5. 评估与验证体系

### 5.1 条件准确率 vs 全局准确率

**新发现**：条件准确率更能反映实际接受率

```python
# 实现：src/speculators/models/metrics.py
def compute_accuracy_single_step(pred_ids, target_ids, loss_mask, prev_correct):
    """计算单步准确率"""
    correct = pred_ids == target_ids
    
    if prev_correct is not None:
        # 条件准确率：只有前面都正确，才计算这一步
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    
    correct_sum = correct.float().sum()
    full_total = torch.tensor(correct.numel(), dtype=torch.float)
    
    return correct_sum, full_total, correct_sum, cond_total
```

**为什么条件准确率重要**：
```
示例：推测 3 个 token
全局准确率：[90%, 80%, 70%]
条件准确率：[90%, 80/90%=89%, 70/80%=87.5%]

实际接受率：
- 全部接受（3个）：0.9 × 0.89 × 0.875 = 70%
- 至少接受 2 个：0.9 × 0.89 = 80%
- 至少接受 1 个：0.9 = 90%

全局准确率会高估实际接受率！
```

### 5.2 DFlash 的位置准确率

**新发现**：DFlash 使用位置准确率评估

```python
# 实现：src/speculators/models/dflash/metrics.py
def compute_metrics(logits, targets, loss_mask, block_size, gamma, loss_fn):
    """计算 DFlash 的指标"""
    pos_idx = torch.arange(seq_len) % block_size
    
    # 计算每个位置的准确率
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    
    metrics = {}
    # 位置 0 是 anchor，不计入准确率
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()
    
    # 每个位置的准确率
    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
    
    return loss, metrics
```

**位置准确率的含义**：
```
Block size = 8
Position 0: anchor（不评估）
Position 1: 第一个预测 token
Position 2: 第二个预测 token
...
Position 7: 第七个预测 token

预期：位置越远，准确率越低
```

### 5.3 Loss Decay 的设计

**新发现**：不同位置的 loss 权重不同

```python
# EAGLE-3 的 loss decay
def exp_loss_decay(pos_idx, gamma):
    """指数衰减：gamma^pos_idx"""
    return gamma ** pos_idx

# DFlash 的 loss decay
def dflash_loss_decay(pos_idx, gamma):
    """DFlash 特殊衰减"""
    # 位置 0 权重为 0（anchor）
    # 位置 1 权重为 1
    # 后续位置指数衰减
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    return decay_mult
```

**为什么使用 loss decay**：
1. **位置重要性**：前面的位置更重要（被接受的概率更高）
2. **训练稳定性**：减少远距离预测的梯度干扰
3. **聚焦关键位置**：让模型专注于高价值的预测

---

## 6. 实际复现中的关键决策

### 6.1 序列长度的选择

**权衡**：
```
长序列（8192+）：
- 优势：覆盖更长的上下文
- 劣势：显存占用大，训练慢

短序列（2048-4096）：
- 优势：显存占用小，训练快
- 劣势：无法覆盖长上下文

推荐：
- 1.5B 模型：4096
- 3B 模型：8192
- 7B 模型：4096（受限于显存）
```

### 6.2 Training Samples 的选择

**权衡**：
```
少量样本（1k-5k）：
- 优势：快速验证流程
- 劣势：模型效果差

大量样本（50k+）：
- 优势：模型效果好
- 劣势：训练时间长

推荐：
- 学习阶段：5k 样本
- 正式训练：50k+ 样本
```

### 6.3 Draft Model 层数的选择

**权衡**：
```
1 层：
- 优势：参数少，训练快
- 劣势：表达能力有限

3-4 层：
- 优势：效果好
- 劣势：参数多，训练慢

推荐：
- 学习阶段：1 层
- 正式训练：3-4 层
```

### 6.4 Learning Rate 的选择

**经验数据**：
```
EAGLE-3：1e-4（默认）
DFlash：3e-4（默认）
小模型：可以尝试更大的 lr（5e-4）
大模型：可以尝试更小的 lr（5e-5）
```

### 6.5 Draft Vocab Size 的选择

**权衡**：
```
小 vocab（16k）：
- 优势：计算量小，训练快
- 劣势：可能丢失一些 token

大 vocab（128k）：
- 优势：覆盖所有 token
- 劣势：计算量大，显存占用大

推荐：
- 学习阶段：16k-32k
- 正式训练：根据 target vocab 大小调整
```

---

## 总结：第三轮新增知识点

### 资源规划
1. **显存预算计算**：模型参数 + 梯度 + 优化器状态 + 激活值
2. **在线 vs 离线选择**：显存约束决定模式选择
3. **模型大小选择**：资源约束下的最优选择

### 训练流程
1. **Multipack Batch Sampler**：LPT 算法提高 GPU 利用率
2. **Noise Transform**：提高模型鲁棒性
3. **Vocabulary Mapping**：减少 draft model 计算量

### 分布式训练
1. **FSDP Mixed Precision**：参数 bf16 + 梯度 fp32
2. **Checkpoint 管理**：分布式保存与加载
3. **Graceful Shutdown**：中断时自动保存

### 数据管线
1. **Hidden States 格式**：safetensors 格式，支持内存映射
2. **Online Training 数据流**：实时生成 + 缓存
3. **Token Frequency**：用于构建 draft vocabulary

### 评估体系
1. **条件准确率**：比全局准确率更能反映实际接受率
2. **位置准确率**：评估不同位置的预测质量
3. **Loss Decay**：不同位置的权重不同

### 关键决策
1. **序列长度**：根据模型大小和显存约束选择
2. **训练样本**：学习阶段少量，正式训练大量
3. **Draft 层数**：学习阶段 1 层，正式训练 3-4 层
4. **Learning Rate**：EAGLE-3 1e-4，DFlash 3e-4
5. **Draft Vocab Size**：16k-32k 适合学习，正式训练根据需求调整
