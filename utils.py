import torch
from torch.utils.data import Dataset, DataLoader

class EnzymeDataset(Dataset):
    """
    扩展后的数据集：除了序列、底物、温度和标签，还包含每个样本的 Cα 原子坐标列表。
    """
    def __init__(self,
                 enzyme_feat: torch.Tensor,
                 substrate_feat: torch.Tensor,
                 temp_embed: torch.Tensor,
                 kcat_labels: torch.Tensor,
                 reaction_flags: torch.Tensor,
                 cluster_ids: torch.Tensor,
                 coords_list: list):
        """
        enzyme_feat: [N, enzyme_dim]
        substrate_feat: [N, substrate_dim]
        temp_embed: [N, 1]
        kcat_labels: [N, 1]
        reaction_flags: [N]
        cluster_ids: [N]
        coords_list: 长度 N 的列表，每个元素为 numpy array 或 Tensor，shape [n_ca, 3]
        """
        self.enzyme_feat = enzyme_feat
        self.substrate_feat = substrate_feat
        self.temp_embed = temp_embed
        self.kcat_labels = kcat_labels
        self.reaction_flags = reaction_flags
        self.cluster_ids = cluster_ids

        # 将 coords_list 转为 Tensor 列表
        self.coords_list = []
        for c in coords_list:
            if isinstance(c, torch.Tensor):
                self.coords_list.append(c)
            else:
                # numpy array 转为 Tensor
                self.coords_list.append(torch.tensor(c, dtype=torch.float32))

    def __len__(self):
        return self.enzyme_feat.size(0)

    def __getitem__(self, idx):
        return (
            self.enzyme_feat[idx],        # [enzyme_dim]
            self.substrate_feat[idx],     # [substrate_dim]
            self.temp_embed[idx],         # [1]
            self.kcat_labels[idx],        # [1]
            self.reaction_flags[idx],     # scalar
            self.cluster_ids[idx],        # scalar
            self.coords_list[idx],        # [n_ca, 3]
        )

def collate_fn(batch):
    """
    自定义 collate，用于处理可变节点数的 coords_list。
    """
    enzyme_feats, substrate_feats, temp_embeds, kcat_labels, reaction_flags, cluster_ids, coords = zip(*batch)
    enzyme_feats = torch.stack(enzyme_feats, dim=0)
    substrate_feats = torch.stack(substrate_feats, dim=0)
    temp_embeds = torch.stack(temp_embeds, dim=0)
    kcat_labels = torch.stack(kcat_labels, dim=0)
    reaction_flags = torch.stack(reaction_flags, dim=0)
    cluster_ids = torch.stack(cluster_ids, dim=0)
    # coords 保持为 list of Tensor
    return enzyme_feats, substrate_feats, temp_embeds, kcat_labels, reaction_flags, cluster_ids, coords

def get_dataloader(dataset: Dataset,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 0):
    """
    DataLoader 使用自定义 collate_fn，以支持 variable-length coords.
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn)
