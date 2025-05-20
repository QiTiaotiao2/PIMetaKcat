import random
import torch
import torch.nn as nn
import numpy as np
from losses import physics_loss
import torch.nn.functional as F
import config
import math

class Trainer:
    def __init__(self, model, optimizer, head_optimizer, device, global_logA, use_classification=False, dataset=None):
        self.model       = model.to(device)
        self.optimizer   = optimizer
        self.device      = device
        self.global_logA = torch.tensor(global_logA, device=device)
        self.dataset     = dataset

    def train_epoch(self,
                    dataloader,
                    current_epoch,
                    total_epochs,
                    lambda_ewc=0.1,
                    use_ewc=False,
                    old_model=None,
                    base_physics_weight=0.2):
        self.model.train()
        total_loss = 0.0
        anneal = min(1.0, current_epoch / (total_epochs / 2))
        physics_weight = base_physics_weight * anneal

        for batch in dataloader:
            (enzyme_feat,
             substrate_feat,
             temp_embed,
             kcat_labels,
             reaction_flags,
             cluster_ids,
             coords_list) = batch

            enzyme_feat    = enzyme_feat.to(self.device)
            substrate_feat = substrate_feat.to(self.device)
            temp_embed     = temp_embed.to(self.device)
            kcat_labels    = kcat_labels.to(self.device).squeeze(-1)
            cluster_ids    = cluster_ids.to(self.device)

            # forward
            preds = self.model(
                enzyme_feat, substrate_feat, temp_embed, coords_list
            ).squeeze(-1)  # [B] or [B, T]

            # 回归 MSE
            loss = nn.MSELoss()(preds, kcat_labels)

            # 物理约束（Arrhenius）
            if temp_embed.size(1) > 0:
                sample_T = temp_embed[:, 0] * 100 + 273.15
                T_ref    = torch.tensor(298.15, device=self.device)
                physL    = physics_loss(preds, sample_T, T_ref,
                                        E_a=50000.0,
                                        logA=self.global_logA,
                                        R=8.314,
                                        weight=physics_weight,
                                        loss_type='L2')
                loss += physL

            # 同簇一致性
            if config.USE_CLUSTER_LOSS:
                for c in torch.unique(cluster_ids):
                    mask = (cluster_ids == c)
                    if mask.sum() > 1:
                        cp = preds[mask]
                        loss += 0.1 * ((cp - cp.mean())**2).mean()

            # backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss
