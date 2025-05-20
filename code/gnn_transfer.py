import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

def build_spatial_graph(coords: torch.Tensor, radius: float = 8.0) -> torch.Tensor:
    """
    基于 3D 坐标构建空间图。
    - coords: Tensor [N_ca, 3]
    - radius: 距离阈值，单位同 PDB 文件坐标（Å）
    返回 edge_index: Tensor [2, E]，无自环。
    """
    N = coords.size(0)
    if N < 2:
        # 无法构图
        return torch.empty((2, 0), dtype=torch.long, device=coords.device)
    # 计算两两距离矩阵
    dist_matrix = torch.cdist(coords, coords)  # [N, N]
    # 找出小于阈值的索引
    src, dst = (dist_matrix < radius).nonzero(as_tuple=True)
    # 排除自环
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    return edge_index

class EnzymeGNN(nn.Module):
    """
    基于 GAT 的 GNN，用于处理蛋白质 Cα 原子图。
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super(EnzymeGNN, self).__init__()
        # 将节点特征（3D 坐标）编码到隐藏空间
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        # 若无边属性，则不使用 edge_encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim) if edge_dim is not None else None
        # GAT 两层
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1)
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N_ca, node_dim] 节点特征（此处为坐标）
        edge_index: [2, E]
        edge_attr: 可选，[E, edge_dim]
        返回 node_embeddings: [N_ca, hidden_dim]
        """
        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        return x

class CrossClusterTransfer(nn.Module):
    """
    新版 CrossClusterTransfer：基于每个样本的 Cα 坐标构建图，
    然后对节点做 GNN，最后池化成每个样本的单个向量。
    """
    def __init__(self,
                 node_dim: int = 3,
                 hidden_dim: int = 128,
                 radius: float = 8.0):
        """
        node_dim: 节点特征维度（此处用 3D 坐标）
        hidden_dim: GNN 隐藏维度
        radius: 构图阈值
        """
        super(CrossClusterTransfer, self).__init__()
        self.radius = radius
        self.gnn = EnzymeGNN(node_dim, edge_dim=None, hidden_dim=hidden_dim)

    def forward(self, coords_list: list) -> torch.Tensor:
        """
        coords_list: 长度 B 的列表，每个元素为 Tensor [N_ca, 3]
        返回：enzyme_embeddings: [B, hidden_dim]
        """
        device = next(self.parameters()).device
        batch_embs = []
        for coords in coords_list:
            coords = coords.to(device)  # [N_ca, 3]
            edge_index = build_spatial_graph(coords, radius=self.radius)
            node_emb = self.gnn(coords, edge_index)  # [N_ca, hidden_dim]
            # 池化所有节点得到样本级表示
            if node_emb.size(0) > 0:
                sample_emb = node_emb.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            else:
                # 若无节点，则全零
                sample_emb = torch.zeros((1, self.gnn.lin.out_features), device=device)
            batch_embs.append(sample_emb)
        # 拼成 [B, hidden_dim]
        enzyme_embeddings = torch.cat(batch_embs, dim=0)
        return enzyme_embeddings
