# 8x4090 复现 Speculators 项目完整计划

> 硬件环境：8x NVIDIA RTX 4090 (24GB VRAM each, 192GB total)
> 目标：完整复现 Speculators 训练流程，掌握 Speculative Decoding 技术

---

## 目录

- [第一部分：资源评估与可行性分析](#第一部分资源评估与可行性分析)
- [第二部分：复现方案设计](#第二部分复现方案设计)
- [第三部分：详细执行计划](#第三部分详细执行计划)
- [第四部分：环境配置指南](#第四部分环境配置指南)
- [第五部分：预期结果与验证](#第五部分预期结果与验证)
- [附录：关键命令与脚本](#附录关键命令与脚本)

---

## 第一部分：资源评估与可行性分析

### 1.1 硬件资源对比

| 资源 | 官方示例 (H100) | 我们的环境 (4090) | 差异分析 |
|------|----------------|------------------|---------|
| **单卡显存** | 80GB | 24GB | 4090 显存约为 H100 的 30% |
| **总显存 (8卡)** | 640GB | 192GB | 可用显存约为官方的 30% |
| **FP16 算力** | 989 TFLOPS | 330 TFLOPS | 4090 算力约为 H100 的 33% |
| **内存带宽** | 3.35 TB/s | 1.0 TB/s | 4090 带宽约为 H100 的 30% |
| **NVLink** | 支持 | 不支持 | 4090 使用 PCIe，通信效率较低 |

### 1.2 模型显存需求分析

**8B 模型（如 Qwen3-8B, Llama3-8B）**：
```
模型参数：8B = 8 × 10^9 参数
FP16 显存需求：8B × 2 bytes = 16GB
BF16 显存需求：8B × 2 bytes = 16GB
INT8 量化：8B × 1 byte = 8GB
INT4 量化：8B × 0.5 byte = 4GB

加上 KV Cache、激活值等开销：
- 推理模式：约 20-25GB（FP16）
- 训练模式：约 40-60GB（需要梯度、优化器状态）
```

**Draft Model（如 EAGLE-3 的 1 层 Transformer）**：
```
参数量：约 100-400M
FP16 显存需求：约 200-800MB
训练显存需求：约 1-2GB
```

### 1.3 可行性评估

**核心挑战**：
1. **8B 模型无法在单卡 4090 上训练**：需要 40-60GB 显存，4090 只有 24GB
2. **vLLM 推理需要足够显存**：8B 模型推理需要约 20-25GB
3. **在线训练需要同时运行 vLLM 和训练**：显存需求翻倍

**解决方案**：

| 方案 | 可行性 | 优势 | 劣势 |
|------|--------|------|------|
| **方案 A：使用小模型** | ⭐⭐⭐⭐⭐ | 完全可行，学习流程 | 模型效果可能不如 8B |
| **方案 B：量化模型** | ⭐⭐⭐⭐ | 可用 8B 模型 | 需要量化支持 |
| **方案 C：离线模式** | ⭐⭐⭐⭐ | 显存效率高 | 需要预生成数据 |
| **方案 D：分布式推理** | ⭐⭐⭐ | 可用 8B 模型 | 配置复杂 |

**推荐方案**：**方案 A + 方案 C 组合**
- 使用 1-3B 小模型进行完整流程学习
- 使用离线模式减少显存压力
- 成功后再尝试 8B 模型（使用量化或分布式）

---

## 第二部分：复现方案设计

### 2.1 方案 A：小模型全流程复现（推荐）

**目标模型选择**：

| 模型 | 参数量 | 显存需求 | 推荐理由 |
|------|--------|---------|---------|
| **Qwen2.5-1.5B** | 1.5B | ~3GB | 最小，完全可行 |
| **Qwen2.5-3B** | 3B | ~6GB | 效果与资源平衡 |
| **Llama-3.2-3B** | 3B | ~6GB | Meta 官方模型 |

**GPU 分配方案**：
```
方案 A1：离线模式（推荐）
- GPU 0-1：vLLM 推理（2 卡，用于生成 hidden states）
- GPU 2-5：训练（4 卡，FSDP 分布式训练）
- GPU 6-7：备用/评估

方案 A2：在线模式
- GPU 0-3：vLLM 推理（4 卡）
- GPU 4-7：训练（4 卡，FSDP 分布式训练）
```

**显存估算**：
```
1.5B 模型：
- vLLM 推理：~5GB/卡（2 卡 = 10GB，足够）
- 训练：~8GB/卡（4 卡 FSDP = 32GB，足够）
- 总需求：~42GB，8 卡 4090 完全满足

3B 模型：
- vLLM 推理：~8GB/卡（2 卡 = 16GB，足够）
- 训练：~12GB/卡（4 卡 FSDP = 48GB，足够）
- 总需求：~64GB，8 卡 4090 完全满足
```

### 2.2 方案 B：8B 模型量化复现

**量化方案**：
```
INT8 量化：
- 模型大小：~8GB
- 推理显存：~12GB（含 KV Cache）
- 训练显存：~24GB（需要梯度检查点）

INT4 量化（GPTQ/AWQ）：
- 模型大小：~4GB
- 推理显存：~8GB
- 训练显存：~16GB
```

**GPU 分配方案**：
```
方案 B1：INT8 量化
- GPU 0-3：vLLM 推理（4 卡，INT8 量化）
- GPU 4-7：训练（4 卡，FSDP + 梯度检查点）

方案 B2：INT4 量化
- GPU 0-1：vLLM 推理（2 卡，INT4 量化）
- GPU 2-7：训练（6 卡，FSDP）
```

### 2.3 方案 C：离线模式复现

**核心思路**：先用 vLLM 生成所有 hidden states，再进行训练

**流程**：
```
Step 1: 数据准备（CPU，~5 分钟）
Step 2: 启动 vLLM，生成 hidden states（~30 分钟）
Step 3: 停止 vLLM，释放显存
Step 4: 使用预生成的 hidden states 训练（~1 小时）
```

**优势**：
- vLLM 和训练不同时运行，显存需求减半
- 可以使用更大的 batch size
- 训练更稳定

**劣势**：
- 需要额外的磁盘空间存储 hidden states
- 无法实时调整数据

---

## 第三部分：详细执行计划

### 3.1 Phase 0：环境准备（Day 1）

**任务清单**：
```markdown
- [ ] 检查 GPU 状态和驱动版本
- [ ] 安装 CUDA 和 cuDNN
- [ ] 创建 Python 虚拟环境
- [ ] 安装 PyTorch（支持 CUDA）
- [ ] 安装 speculators 和 vLLM
- [ ] 验证安装
```

**关键命令**：
```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 创建虚拟环境
conda create -n speculators python=3.10
conda activate speculators

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 speculators
pip install -e ".[dev]"

# 安装 vLLM
pip install vllm

# 验证安装
python -c "import speculators; print(speculators.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

### 3.2 Phase 1：小模型快速验证（Day 2-3）

**目标**：使用 1.5B 模型验证完整流程

**Step 1.1：数据准备**
```bash
# 准备 ShareGPT 数据集（5k 样本）
python scripts/prepare_data.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data sharegpt \
    --output ./output/qwen2_1_5b \
    --max-samples 5000 \
    --seq-length 4096  # 使用较短序列减少显存
```

**Step 1.2：离线生成 Hidden States**
```bash
# 启动 vLLM（使用 2 卡）
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py Qwen/Qwen2.5-1.5B-Instruct \
    -- --data-parallel-size 2 --port 8000 &

# 等待 vLLM 启动
sleep 60

# 生成 hidden states
python scripts/data_generation_offline.py \
    --preprocessed-data ./output/qwen2_1_5b \
    --endpoint http://localhost:8000/v1 \
    --output ./output/qwen2_1_5b/hidden_states \
    --max-samples 5000 \
    --concurrency 32 \
    --validate-outputs

# 停止 vLLM
kill %1
```

**Step 1.3：训练 EAGLE-3**
```bash
# 使用 4 卡训练
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun \
    --standalone --nproc_per_node 4 \
    scripts/train.py \
    --verifier-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
    --data-path ./output/qwen2_1_5b \
    --hidden-states-path ./output/qwen2_1_5b/hidden_states \
    --save-path ./output/qwen2_1_5b/checkpoints \
    --draft-vocab-size 16000 \
    --epochs 5 \
    --lr 1e-4 \
    --total-seq-len 4096 \
    --num-layers 1 \
    --on-missing raise
```

**Step 1.4：评估**
```bash
# 使用 vLLM 加载训练好的 draft model
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-model ./output/qwen2_1_5b/checkpoints/checkpoint_best \
    --num-speculative-tokens 3 \
    --port 8000

# 运行评估
python scripts/evaluate/evaluate.py \
    --endpoint http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --benchmark mt_bench \
    --num-prompts 80
```

### 3.3 Phase 2：3B 模型扩展验证（Day 4-5）

**目标**：验证更大模型的可行性

**Step 2.1：数据准备**
```bash
# 使用更长的序列
python scripts/prepare_data.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data sharegpt \
    --output ./output/qwen2_3b \
    --max-samples 10000 \
    --seq-length 8192
```

**Step 2.2：离线生成 Hidden States**
```bash
# 使用 4 卡 vLLM（3B 模型需要更多显存）
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py Qwen/Qwen2.5-3B-Instruct \
    -- --data-parallel-size 4 --port 8000 &

# 生成 hidden states
python scripts/data_generation_offline.py \
    --preprocessed-data ./output/qwen2_3b \
    --endpoint http://localhost:8000/v1 \
    --output ./output/qwen2_3b/hidden_states \
    --max-samples 10000 \
    --concurrency 16 \
    --validate-outputs
```

**Step 2.3：训练多种算法**
```bash
# 训练 EAGLE-3
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone --nproc_per_node 4 \
    scripts/train.py \
    --verifier-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --data-path ./output/qwen2_3b \
    --hidden-states-path ./output/qwen2_3b/hidden_states \
    --save-path ./output/qwen2_3b/eagle3_checkpoints \
    --speculator-type eagle3 \
    --draft-vocab-size 32000 \
    --epochs 5 \
    --lr 1e-4 \
    --total-seq-len 8192

# 训练 DFlash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone --nproc_per_node 4 \
    scripts/train.py \
    --verifier-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --data-path ./output/qwen2_3b \
    --hidden-states-path ./output/qwen2_3b/hidden_states \
    --save-path ./output/qwen2_3b/dflash_checkpoints \
    --speculator-type dflash \
    --draft-vocab-size 32000 \
    --epochs 5 \
    --lr 3e-4 \
    --total-seq-len 8192 \
    --block-size 8 \
    --max-anchors 1024 \
    --num-layers 3
```

### 3.4 Phase 3：8B 模型挑战（Day 6-7）

**目标**：尝试使用 8B 模型，探索显存优化

**方案 3A：INT8 量化**
```bash
# 使用 INT8 量化加载 vLLM
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py Qwen/Qwen2.5-7B-Instruct \
    -- --data-parallel-size 4 --port 8000 --quantization awq &

# 生成 hidden states（使用较短序列）
python scripts/data_generation_offline.py \
    --preprocessed-data ./output/qwen2_7b \
    --endpoint http://localhost:8000/v1 \
    --output ./output/qwen2_7b/hidden_states \
    --max-samples 5000 \
    --concurrency 8 \
    --validate-outputs
```

**方案 3B：梯度检查点训练**
```bash
# 使用梯度检查点减少显存
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone --nproc_per_node 4 \
    scripts/train.py \
    --verifier-name-or-path Qwen/Qwen2.5-7B-Instruct \
    --data-path ./output/qwen2_7b \
    --hidden-states-path ./output/qwen2_7b/hidden_states \
    --save-path ./output/qwen2_7b/checkpoints \
    --draft-vocab-size 32000 \
    --epochs 3 \
    --lr 1e-4 \
    --total-seq-len 4096 \
    --gradient-checkpointing \
    --batch-size 1
```

### 3.5 Phase 4：在线训练模式（Day 8-9）

**目标**：验证在线训练流程

**在线训练脚本**：
```bash
# 分配 GPU：0-3 给 vLLM，4-7 给训练
VLLM_GPUS="0,1,2,3"
TRAIN_GPUS="4,5,6,7"

# 启动 vLLM
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py Qwen/Qwen2.5-3B-Instruct \
    -- --data-parallel-size 4 --port 8000 &

# 等待 vLLM 启动
sleep 60

# 在线训练
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node 4 \
    scripts/train.py \
    --verifier-name-or-path Qwen/Qwen2.5-3B-Instruct \
    --data-path ./output/qwen2_3b \
    --vllm-endpoint http://localhost:8000/v1 \
    --save-path ./output/qwen2_3b/online_checkpoints \
    --draft-vocab-size 32000 \
    --epochs 5 \
    --lr 1e-4 \
    --total-seq-len 8192 \
    --on-missing generate \
    --on-generate delete
```

### 3.6 Phase 5：完整评估与对比（Day 10）

**评估指标**：
```bash
# 评估 acceptance rate
python scripts/evaluate/evaluate.py \
    --endpoint http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --benchmark mt_bench \
    --num-prompts 80

# 评估延迟和吞吐量
python scripts/evaluate/perf_utils.py \
    --endpoint http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-requests 100 \
    --max-tokens 256
```

**对比实验**：
```
实验 1：不同模型大小（1.5B vs 3B vs 7B）
实验 2：不同算法（EAGLE-3 vs DFlash）
实验 3：不同训练数据量（1k vs 5k vs 10k）
实验 4：不同推测长度（K=3 vs K=5 vs K=7）
```

---

## 第四部分：环境配置指南

### 4.1 软件环境

**操作系统**：
```bash
# 推荐 Ubuntu 22.04 LTS
lsb_release -a
```

**CUDA 版本**：
```bash
# 检查 CUDA 版本
nvcc --version

# 推荐 CUDA 12.1+
# 如果需要安装 CUDA：
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

**Python 环境**：
```bash
# 使用 Conda 管理环境
conda create -n speculators python=3.10
conda activate speculators

# 或使用 venv
python -m venv .venv
source .venv/bin/activate
```

### 4.2 依赖安装

**PyTorch 安装**：
```bash
# 根据 CUDA 版本选择
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"
```

**speculators 安装**：
```bash
# 克隆仓库
git clone https://github.com/vllm-project/speculators.git
cd speculators

# 安装
pip install -e ".[dev]"

# 验证
speculators --version
```

**vLLM 安装**：
```bash
# 安装 vLLM
pip install vllm

# 或从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# 验证
python -c "import vllm; print(vllm.__version__)"
```

### 4.3 模型下载

**使用 HuggingFace Hub**：
```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 登录（如果需要访问私有模型）
huggingface-cli login

# 下载模型
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='./models/qwen2_5-1_5b')
"
```

**使用 modelscope（国内镜像）**：
```bash
# 安装 modelscope
pip install modelscope

# 下载模型
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ./models/qwen2_5-1_5b
```

### 4.4 数据集准备

**ShareGPT 数据集**：
```bash
# 自动下载（speculators 内置支持）
python scripts/prepare_data.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data sharegpt \
    --output ./data/sharegpt \
    --max-samples 5000
```

**自定义数据集**：
```bash
# 准备 JSONL 格式数据
# 格式：{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

python scripts/prepare_data.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data ./data/custom.jsonl \
    --output ./data/custom \
    --max-samples 5000
```

---

## 第五部分：预期结果与验证

### 5.1 预期结果

**1.5B 模型（5k 样本训练）**：
```
接受率：10-15%
接受长度：1.3-1.5
位置接受率：
  - 位置 0：30-40%
  - 位置 1：5-10%
  - 位置 2：1-3%
输出吞吐量：100-200 tok/s
```

**3B 模型（10k 样本训练）**：
```
接受率：15-20%
接受长度：1.4-1.6
位置接受率：
  - 位置 0：35-45%
  - 位置 1：8-12%
  - 位置 2：2-5%
输出吞吐量：150-250 tok/s
```

### 5.2 验证检查点

**Checklist**：
```markdown
## 环境验证
- [ ] GPU 状态正常（nvidia-smi）
- [ ] CUDA 版本正确
- [ ] PyTorch CUDA 支持正常
- [ ] speculators 安装成功
- [ ] vLLM 安装成功

## 数据验证
- [ ] 数据集下载成功
- [ ] 数据预处理完成
- [ ] Hidden states 生成成功
- [ ] 数据格式正确

## 训练验证
- [ ] 训练启动成功
- [ ] Loss 正常下降
- [ ] 准确率提升
- [ ] Checkpoint 保存成功

## 评估验证
- [ ] vLLM 加载 draft model 成功
- [ ] 推理正常工作
- [ ] 接受率符合预期
- [ ] 延迟有改善
```

### 5.3 常见问题与解决方案

**问题 1：CUDA Out of Memory**
```
症状：RuntimeError: CUDA out of memory
解决方案：
1. 减少 batch size
2. 减少序列长度
3. 使用梯度检查点
4. 使用更小的模型
5. 使用量化
```

**问题 2：vLLM 启动失败**
```
症状：vLLM server 无法启动
解决方案：
1. 检查端口是否被占用
2. 检查 GPU 显存是否足够
3. 检查模型路径是否正确
4. 查看 vLLM 日志
```

**问题 3：训练 Loss 不下降**
```
症状：Loss 震荡或停滞
解决方案：
1. 调整学习率
2. 检查数据质量
3. 增加训练数据量
4. 调整模型架构
```

**问题 4：NCCL 通信错误**
```
症状：分布式训练失败
解决方案：
1. 检查 NCCL 版本
2. 设置 NCCL 环境变量
3. 减少 GPU 数量
4. 使用 gloo 后端
```

---

## 附录：关键命令与脚本

### A.1 快速启动脚本

**train_eagle3_small.sh**：
```bash
#!/bin/bash
# 小模型 EAGLE-3 训练脚本

set -euo pipefail

# 配置
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="./output/qwen2_1_5b_eagle3"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=1e-4

# GPU 分配
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3,4,5"
NUM_TRAIN_GPUS=4

# Step 1: 数据准备
echo "=== Step 1: 数据准备 ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data sharegpt \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: 启动 vLLM
echo "=== Step 2: 启动 vLLM ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "停止 vLLM..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "等待 vLLM 启动..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM 已启动"

# Step 3: 生成 hidden states
echo "=== Step 3: 生成 hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$OUTPUT_DIR/hidden_states" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency 32 \
    --validate-outputs

# Step 4: 停止 vLLM
echo "=== Step 4: 停止 vLLM ==="
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

# Step 5: 训练
echo "=== Step 5: 训练 ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size 16000 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

echo "完成！Checkpoint 保存在 $OUTPUT_DIR/checkpoints/"
```

### A.2 评估脚本

**evaluate_model.sh**：
```bash
#!/bin/bash
# 评估脚本

set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT="./output/qwen2_1_5b_eagle3/checkpoints/checkpoint_best"
VLLM_PORT=8000

# 启动 vLLM with speculative decoding
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --speculative-model "$CHECKPOINT" \
    --num-speculative-tokens 3 \
    --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 60

# 运行评估
python scripts/evaluate/evaluate.py \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --model "$MODEL" \
    --benchmark mt_bench \
    --num-prompts 80
```

### A.3 监控脚本

**monitor_gpu.sh**：
```bash
#!/bin/bash
# GPU 监控脚本

while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv
    echo ""
    sleep 5
done
```

### A.4 环境检查脚本

**check_env.sh**：
```bash
#!/bin/bash
# 环境检查脚本

echo "=== 系统信息 ==="
uname -a

echo ""
echo "=== GPU 信息 ==="
nvidia-smi

echo ""
echo "=== CUDA 版本 ==="
nvcc --version

echo ""
echo "=== Python 版本 ==="
python --version

echo ""
echo "=== PyTorch 版本 ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "=== speculators 版本 ==="
python -c "import speculators; print(speculators.__version__)"

echo ""
echo "=== vLLM 版本 ==="
python -c "import vllm; print(vllm.__version__)"
```

---

## 总结

### 推荐路径

```
Day 1: 环境准备
Day 2-3: 1.5B 模型全流程验证
Day 4-5: 3B 模型扩展验证
Day 6-7: 8B 模型挑战（可选）
Day 8-9: 在线训练模式验证
Day 10: 完整评估与总结
```

### 关键成功因素

1. **从小模型开始**：先用 1.5B 验证流程，再扩展到更大模型
2. **使用离线模式**：减少显存压力，提高稳定性
3. **监控资源使用**：及时发现和解决问题
4. **记录实验结果**：便于对比和优化

### 预期收获

1. **技术掌握**：
   - 理解 Speculative Decoding 原理
   - 掌握 EAGLE-3、DFlash 等算法
   - 熟悉分布式训练流程

2. **工程能力**：
   - GPU 资源管理
   - 训练流程优化
   - 问题排查能力

3. **面试准备**：
   - 真实项目经验
   - 可展示的实验结果
   - 深入的技术理解
