#!/bin/bash

# 遇到错误立即停止脚本
set -e

# ================= 配置区域 =================
# 定义基础路径
BASE_HDD="/inspire/hdd/project/robot-body/yanglixin-p-yanglixin/jialin"
BASE_SSD="/inspire/ssd/project/robot-body/yanglixin-p-yanglixin/jialin"
CONDA_PATH="${BASE_HDD}/miniconda3"

# 预训练模型路径 (请修改为您真实的 .pt 文件路径)
# 例如: 
PRETRAIN_CKPT="${BASE_HDD}/UMAP/workspace/logs/uwm/libero_90/pretrain_6_offline/0/models.pt"

# ================= 1. 环境初始化 =================
echo ">>> Initializing Conda..."
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate uwm

# ================= 2. 设置环境变量 =================
echo ">>> Exporting Environment Variables..."

# 强制设置单机训练的主节点地址 (防止 NCCL 连错 IP)
export MASTER_ADDR=localhost
export MASTER_PORT=29501  # 使用不同端口，避免和训练任务冲突

export LOG_DIR="${BASE_HDD}/UMAP/workspace/logs"
export LIBERO_ROOT="/inspire/hdd/project/robot-body/public/jltian/libero_data"
export LIBERO_BUFFER_ROOT="${BASE_SSD}/data/uwm_offline"
export PYTHONPATH="${BASE_HDD}/UMAP/dependencies/uwm_motion:$PYTHONPATH"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

# ================= 3. 准备工作 =================
WORK_DIR="${BASE_HDD}/UMAP/dependencies/uwm_motion"
cd "${WORK_DIR}"
echo ">>> Working Directory: $(pwd)"

# ================= 4. 开始微调 =================
echo ">>> Starting Finetuning..."

# 注意：如果您的环境是 4机8卡(32 GPU)，这里必须设置 batch_size=4 以保持 Global Batch Size 合理
# 如果是 1机4卡，可以去掉 batch_size=4 或者设为 36

HYDRA_FULL_ERROR=1 NCCL_DEBUG=INFO \
python experiments/uwm/train_robomimic.py \
    --config-name finetune_uwm_robomimic.yaml \
    dataset=libero_book_caddy \
    exp_id=finetune_test \
    pretrain_checkpoint_path="${PRETRAIN_CKPT}"

echo ">>> Finetuning script finished."
