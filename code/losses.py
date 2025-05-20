# # import torch
# # import torch.nn.functional as F
# #
# # def physics_loss(kcat_pred, temperature, T_ref, E_a, logA, R=8.314, weight=1.0, loss_type='L2'):
# #     """
# #     基于 Arrhenius 方程约束模型预测的 log₁₀(kcat)：
# #       期望：log₁₀(kcat) = logA - (E_a/(R*ln10))*(1/T - 1/T_ref)
# #     参数：
# #       kcat_pred: 模型预测的 log₁₀(kcat)，形状 [batch_size, 1] 或 [batch_size]
# #       temperature: 每个样本的温度（单位 Kelvin），形状 [batch_size]
# #       T_ref: 参考温度（单位 Kelvin，如 298.15），标量张量
# #       E_a: 活化能（单位 J/mol）
# #       logA: 参考温度下的基准 log₁₀(kcat)，标量张量（建议由训练数据均值确定）
# #       R: 气体常数，默认 8.314
# #       weight: 物理损失权重
# #       loss_type: 'L1' 或 'L2'，默认 'L2'
# #     """
# #     ln10 = torch.log(torch.tensor(10.0, device=kcat_pred.device))
# #     expected = logA - (E_a / (R * ln10)) * (1.0 / temperature - 1.0 / T_ref)
# #     if loss_type == 'L1':
# #         loss = torch.mean(torch.abs(kcat_pred.squeeze() - expected))
# #     elif loss_type == 'L2':
# #         loss = torch.mean((kcat_pred.squeeze() - expected) ** 2)
# #     else:
# #         loss = torch.mean(torch.abs(kcat_pred.squeeze() - expected))
# #     return weight * loss
# #
# # def ewc_loss(model, old_params, fisher_info, lambda_ewc):
# #     loss = 0.0
# #     for name, param in model.named_parameters():
# #         if name in old_params:
# #             loss += (fisher_info[name] * (param - old_params[name]).pow(2)).sum()
# #     return lambda_ewc * loss
# #
# # def alignment_loss(feat_A, feat_B):
# #     cos_sim = F.cosine_similarity(feat_A, feat_B, dim=1)
# #     return (1 - cos_sim.mean())
# #
# # def distillation_loss(new_output, old_output, temperature=2.0):
# #     log_probs = F.log_softmax(new_output / temperature, dim=1)
# #     probs_old = F.softmax(old_output / temperature, dim=1)
# #     loss = F.kl_div(log_probs, probs_old, reduction='batchmean') * (temperature ** 2)
# #     return loss
# # losses.py
# import torch
# import torch.nn.functional as F
#
#
# def physics_loss(kcat_pred, temperature, T_ref, E_a, logA, R=8.314, weight=1.0, loss_type='L2'):
#     """
#     基于 Arrhenius 方程计算物理损失：
#       期望 log₁₀(kcat) = logA - (E_a/(R*ln10)) * (1/T - 1/T_ref)
#
#     参数：
#       kcat_pred: 模型预测的 log₁₀(kcat)，形状为 [N, T, 1]，其中 T 为不同温度点
#       temperature: 每个温度点的温度，形状 [N, T]，单位 Kelvin
#       T_ref: 参考温度（标量张量）
#       E_a: 活化能
#       logA: 参考温度下的基准 log₁₀(kcat)
#       R: 气体常数
#       weight: 物理损失的权重
#       loss_type: 'L1' 或 'L2'
#     """
#     ln10 = torch.log(torch.tensor(10.0, device=kcat_pred.device))
#     # 计算每个温度点的期望值，得到形状 [N, T]
#     expected = logA - (E_a / (R * ln10)) * (1.0 / temperature - 1.0 / T_ref)
#     expected = expected.unsqueeze(-1)  # 形状变为 [N, T, 1]
#     if loss_type == 'L1':
#         loss = torch.abs(kcat_pred - expected).mean()
#     else:  # 默认 L2 损失
#         loss = ((kcat_pred - expected) ** 2).mean()
#     return weight * loss
#
# def ewc_loss(model, old_params, fisher_info, lambda_ewc):
#     loss = 0.0
#     for name, param in model.named_parameters():
#         if name in old_params:
#             loss += (fisher_info[name] * (param - old_params[name]).pow(2)).sum()
#     return lambda_ewc * loss
#
# def alignment_loss(feat_A, feat_B):
#     cos_sim = F.cosine_similarity(feat_A, feat_B, dim=1)
#     return (1 - cos_sim.mean())
#
# def distillation_loss(new_output, old_output, temperature=2.0):
#     log_probs = F.log_softmax(new_output / temperature, dim=1)
#     probs_old = F.softmax(old_output / temperature, dim=1)
#     loss = F.kl_div(log_probs, probs_old, reduction='batchmean') * (temperature ** 2)
#     return loss
import torch
import torch.nn.functional as F


def physics_loss(kcat_pred, temperature, T_ref, E_a, logA, R=8.314, weight=1.0, loss_type='L2'):
    """
    基于 Arrhenius 方程计算物理损失：
      期望 log₁₀(kcat) = logA - (E_a/(R*ln10)) * (1/T - 1/T_ref)

    参数：
      kcat_pred: 模型预测的 log₁₀(kcat)，形状为 [N, T, 1]，T 为不同温度点
      temperature: 每个温度点的温度，形状 [N, T]（单位 Kelvin）
      T_ref: 参考温度（标量张量）
      E_a: 活化能
      logA: 参考温度下的基准 log₁₀(kcat)
      R: 气体常数
      weight: 物理损失的权重
      loss_type: 'L1' 或 'L2'
    """
    ln10 = torch.log(torch.tensor(10.0, device=kcat_pred.device))
    # 计算每个温度点的期望值，形状 [N, T]
    expected = logA - (E_a / (R * ln10)) * (1.0 / temperature - 1.0 / T_ref)
    expected = expected.unsqueeze(-1)  # 转换为 [N, T, 1]

    if loss_type == 'L1':
        loss = torch.abs(kcat_pred - expected).mean()
    else:  # 默认采用 L2 损失
        loss = ((kcat_pred - expected) ** 2).mean()
    return weight * loss


def ewc_loss(model, old_params, fisher_info, lambda_ewc):
    loss = 0.0
    for name, param in model.named_parameters():
        if name in old_params:
            loss += (fisher_info[name] * (param - old_params[name]).pow(2)).sum()
    return lambda_ewc * loss


def alignment_loss(feat_A, feat_B):
    cos_sim = F.cosine_similarity(feat_A, feat_B, dim=1)
    return (1 - cos_sim.mean())


def distillation_loss(new_output, old_output, temperature=2.0):
    log_probs = F.log_softmax(new_output / temperature, dim=1)
    probs_old = F.softmax(old_output / temperature, dim=1)
    loss = F.kl_div(log_probs, probs_old, reduction='batchmean') * (temperature ** 2)
    return loss
