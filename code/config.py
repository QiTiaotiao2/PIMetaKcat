# # config.py
# import torch
#
# TRAIN_DATA_PATH = "data/DLTKcat_Me_training_final.csv"
# TEST_DATA_PATH = "data/DLTKcat_Me_testing_final2.csv"
#
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# if torch.cuda.is_available():
#     print("使用 GPU 进行训练, GPU 名称:", torch.cuda.get_device_name(0))
# else:
#     raise SystemError("GPU 不可用，请检查CUDA设置")
#
# ENZYME_DIM = 1024
# SUBSTRATE_DIM = 256
# TEMP_DIM = 1  # 仅使用归一化温度标量
# HIDDEN_DIM = 512
# OUTPUT_DIM = 256
#
# BATCH_SIZE = 32
# NUM_EPOCHS = 100
# LEARNING_RATE_GLOBAL = 3e-4  # 降低学习率以稳定训练5
# WEIGHT_PHYSICS = 0.1  # 增强物理约束
# LAMBDA_EWC = 0.1
# USE_EWC = False
#
# META_BATCH_SIZE = 4
# K_SHOT = 5
# USE_META = True
# META_INNER_LR = 0.005  # 调整内循环学习率
# META_META_LR = 0.001
# META_N_STEPS = 3  # 减少步数以避免过拟合
# META_LAMBDA = 0.1
# META_TASK_LIMIT = 8  # 增加任务数量
#
# USE_TEMPERATURE = True
# USE_CLUSTER_LOSS = True
# USE_GNN_TRANSFER = True
#
# USE_PRECOMPUTED_ENZYME = True
# ENZYME_EMBEDDING_PATH_TRAIN = "precomputed/enzyme_embeddings_train.npy"
# ENZYME_EMBEDDING_PATH_TEST = "precomputed/enzyme_embeddings_test.npy"
#
# # 使用基于学习基函数的 few-shot 回归方法
# USE_FEW_SHOT_REGRESSION = True
# FEATURE_DIM = 128  # 基函数数量
#
# # 新增两阶段训练参数
# STAGE1_EPOCHS = 50  # 第一阶段：使用真实任务标签，冻结 TaskLabelGenerator
# STAGE2_EPOCHS = NUM_EPOCHS - STAGE1_EPOCHS  # 第二阶段：解冻 TaskLabelGenerator，采用生成的任务标签
#
import torch

# 训练/测试数据路径
TRAIN_DATA_PATH = "data/DLTKcat_Me_training_final.csv"
TEST_DATA_PATH = "data/DLTKcat_Me_testing_final2.csv"

# 新增：PDB 目录
TRAIN_PDB_DIR = r"D:\StudySpace\paper\Kcat\downloaded_pdbs_train"
TEST_PDB_DIR  = r"D:\StudySpace\paper\Kcat\downloaded_pdbs"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("使用 GPU 进行训练, GPU 名称:", torch.cuda.get_device_name(0))
else:
    raise SystemError("GPU 不可用，请检查CUDA设置")

# （以下参数保持不变）…
ENZYME_DIM = 1024
SUBSTRATE_DIM = 256
TEMP_DIM = 1
HIDDEN_DIM = 512
OUTPUT_DIM = 256

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE_GLOBAL = 5e-4
WEIGHT_PHYSICS = 0.1
LAMBDA_EWC = 0.1
USE_EWC = False

META_BATCH_SIZE = 4
K_SHOT = 5
USE_META = True
META_INNER_LR = 0.005
META_META_LR = 0.001
META_N_STEPS = 3
META_LAMBDA = 0.1
META_TASK_LIMIT = 8

USE_TEMPERATURE = True
USE_CLUSTER_LOSS = True
USE_GNN_TRANSFER = True

USE_PRECOMPUTED_ENZYME = True
ENZYME_EMBEDDING_PATH_TRAIN = "precomputed/enzyme_embeddings_train.npy"
ENZYME_EMBEDDING_PATH_TEST  = "precomputed/enzyme_embeddings_test.npy"

USE_FEW_SHOT_REGRESSION = True
FEATURE_DIM = 128

STAGE1_EPOCHS = 30
STAGE2_EPOCHS = NUM_EPOCHS - STAGE1_EPOCHS
