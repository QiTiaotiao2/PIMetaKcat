import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from gnn_transfer import CrossClusterTransfer

class GlobalEncoder(nn.Module):
    def __init__(self,
                 enzyme_dim: int,
                 substrate_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float = 0.3):
        super(GlobalEncoder, self).__init__()
        self.enzyme_branch = nn.Sequential(
            nn.Linear(enzyme_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        self.substrate_branch = nn.Sequential(
            nn.Linear(substrate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )
        fusion_input_dim = hidden_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(),
        )

    def forward(self,
                enzyme_feat: torch.Tensor,
                substrate_feat: torch.Tensor) -> torch.Tensor:
        e_out = self.enzyme_branch(enzyme_feat)
        s_out = self.substrate_branch(substrate_feat)
        combined = torch.cat([e_out, s_out], dim=1)
        return self.fusion_layer(combined)


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_unsq = x.unsqueeze(1)
        attn_output, _ = self.self_attn(x_unsq, x_unsq, x_unsq)
        x2 = attn_output.squeeze(1)
        x = self.norm1(x2 + residual)
        residual2 = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.norm2(x + residual2)


class EnhancedWeightsGenerator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_blocks: int = 3):
        super(EnhancedWeightsGenerator, self).__init__()
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=hidden_dim,
                             nhead=4,
                             dim_feedforward=hidden_dim // 2)
            for _ in range(num_blocks)
        ])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        return self.fc_out(x)


class TaskLabelGenerator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 task_label_dim: int,
                 num_blocks: int = 3):
        super(TaskLabelGenerator, self).__init__()
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=hidden_dim,
                             nhead=4,
                             dim_feedforward=hidden_dim // 2)
            for _ in range(num_blocks)
        ])
        self.fc_out = nn.Linear(hidden_dim, task_label_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_initial(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=0, keepdim=True)
        return self.sigmoid(self.fc_out(x))


class EnhancedFewShotRegressionModel(nn.Module):
    def __init__(self,
                 enzyme_dim: int,
                 substrate_dim: int,
                 feature_dim: int = config.FEATURE_DIM,
                 task_label_dim: int = 1,
                 use_true_task_labels: bool = True):
        super(EnhancedFewShotRegressionModel, self).__init__()
        # 序列+底物基函数
        self.feature_extractor = GlobalEncoder(
            enzyme_dim=enzyme_dim,
            substrate_dim=substrate_dim,
            hidden_dim=2 * feature_dim,
            output_dim=feature_dim,
            dropout=0.3
        )
        # 结构 GNN
        self.structure_encoder = CrossClusterTransfer(
            node_dim=3,
            hidden_dim=feature_dim,
            radius=8.0
        )
        # 元学习任务权重
        self.use_true_task_labels = use_true_task_labels
        self.task_label_generator = TaskLabelGenerator(
            input_dim=feature_dim + 1,
            hidden_dim=feature_dim // 2,
            task_label_dim=task_label_dim
        )
        self.weights_generator = EnhancedWeightsGenerator(
            input_dim=feature_dim + 1 + task_label_dim,
            hidden_dim=feature_dim // 2,
            output_dim=feature_dim
        )
        # 多层 MLP 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1)
        )

    def compute_task_weights(self,
                             support_enzyme_feat: torch.Tensor,
                             support_substrate_feat: torch.Tensor,
                             support_label: torch.Tensor,
                             task_true_label: torch.Tensor = None) -> torch.Tensor:
        support_features = self.feature_extractor(
            support_enzyme_feat, support_substrate_feat
        )  # [K, feature_dim]
        if self.use_true_task_labels:
            assert task_true_label is not None
            task_label = task_true_label
        else:
            tl_input = torch.cat([support_features, support_label], dim=1)
            task_label = self.task_label_generator(tl_input)
        K = support_features.size(0)
        task_label_rep = task_label.expand(K, -1)
        wg_input = torch.cat([support_features, support_label, task_label_rep], dim=1)
        weight_vectors = self.weights_generator(wg_input)
        return weight_vectors.mean(dim=0, keepdim=True)

    def forward(self,
                enzyme_feat: torch.Tensor,
                substrate_feat: torch.Tensor,
                temp_embed: torch.Tensor,
                coords_list: list) -> torch.Tensor:
        seq_feat   = self.feature_extractor(enzyme_feat, substrate_feat)  # [B, feature_dim]
        struct_feat = self.structure_encoder(coords_list)                # [B, feature_dim]
        fused      = seq_feat + struct_feat                              # [B, feature_dim]
        reg_out    = self.regression_head(fused)                         # [B, 1]
        if temp_embed is not None:
            T = temp_embed.size(1)
            reg_out = reg_out.unsqueeze(1).repeat(1, T, 1)
        return reg_out


# 别名
FewShotRegressionModel = EnhancedFewShotRegressionModel
GlobalEncoder = GlobalEncoder
