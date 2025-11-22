#!/bin/bash

# 遇到错误立即停止脚本
set -e

# ================= 配置区域 =================
# 定义基础路径，方便修改
BASE_HDD="/inspire/hdd/project/robot-body/yanglixin-p-yanglixin/jialin"
BASE_SSD="/inspire/ssd/project/robot-body/yanglixin-p-yanglixin/jialin"
CONDA_PATH="${BASE_HDD}/miniconda3"

# ================= 1. 环境初始化 =================
echo ">>> Initializing Conda..."
# 在脚本中激活 conda 需要先 source conda.sh
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate uwm

# ================= 2. 设置环境变量 =================
echo ">>> Exporting Environment Variables..."

# 日志目录 (建议：如果 HDD 写入太慢导致卡顿，可以考虑改到 SSD)
export LOG_DIR="${BASE_HDD}/UMAP/workspace/logs"

# 数据集路径
export LIBERO_ROOT="/inspire/hdd/project/robot-body/public/jltian/libero_data"

# 缓存/Buffer 路径 (SSD)
export LIBERO_BUFFER_ROOT="${BASE_SSD}/data/uwm_offline"

# Python 路径
export PYTHONPATH="${BASE_HDD}/UMAP/dependencies/uwm_motion:$PYTHONPATH"

# 强制 WandB 离线 (防止网络问题导致训练卡住)
export WANDB_MODE=offline

# 解决 HuggingFace 连接问题 (使用离线模式，前提是模型已下载)
export HF_HUB_OFFLINE=1

# ================= 3. 准备工作 =================
# 切换到工作目录
WORK_DIR="${BASE_HDD}/UMAP/dependencies/uwm_motion"
cd "${WORK_DIR}"
echo ">>> Working Directory: $(pwd)"

# 清理可能残留的僵尸进程 (可选，防止上次训练的 worker 没关掉)
echo ">>> Cleaning up potential zombie processes..."
pkill -f train_robomimic.py || true

# ================= 4. 开始训练 =================
echo ">>> Starting Training..."

# 之前的诊断建议：
# 为了防止 NCCL Timeout 和内存泄漏，建议在参数中显式添加 num_workers 和 persistent_workers 设置
# 如果您已经在 yaml 里改好了，可以去掉下面的 overrides

HYDRA_FULL_ERROR=1 NCCL_DEBUG=INFO \
python experiments/uwm/train_robomimic.py \
    --config-name train_uwm_robomimic.yaml \
    dataset=libero_90 \
    exp_id=pretrain_5_offline \
    num_workers=4 \
    persistent_workers=False

echo ">>> Training script finished."