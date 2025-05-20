# # # # meta_learning.py
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from copy import deepcopy
# # #
# # # # ----------------------------
# # # # FewShotRegression：针对少样本回归任务的元学习模块
# # # # 该模块支持两阶段训练：
# # # #   - Stage 1：使用真实任务标签（簇编号）训练特征提取器和权重生成器；
# # # #   - Stage 2：冻结前两部分参数，更新 TaskLabelGenerator，并用其自动生成任务标签训练整个模型。
# # # # ----------------------------
# # # class FewShotRegression:
# # #     def __init__(self, model, meta_lr=0.001):
# # #         """
# # #         model: EnhancedFewShotRegressionModel 实例，包含 feature_extractor、weights_generator 和 task_label_generator
# # #         meta_lr: 元学习优化率
# # #         """
# # #         self.model = model
# # #         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
# # #         self.mse_loss = nn.MSELoss()
# # #
# # #     def meta_train_step(self, tasks, training_stage=1):
# # #         """
# # #         tasks: 列表，每个任务为 (support_batch, query_batch, task_weight)
# # #         每个 batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
# # #         training_stage: 1 表示第一阶段（使用真实任务标签），2 表示第二阶段（使用生成的任务标签）
# # #         """
# # #         meta_loss = 0.0
# # #         num_tasks = len(tasks)
# # #         # 根据训练阶段设置模型的标志
# # #         if training_stage == 1:
# # #             self.model.use_true_task_labels = True
# # #         else:
# # #             self.model.use_true_task_labels = False
# # #
# # #         for task in tasks:
# # #             if len(task) == 3:
# # #                 support_batch, query_batch, task_weight = task
# # #             else:
# # #                 support_batch, query_batch = task
# # #                 task_weight = 1.0
# # #
# # #             # support_batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
# # #             enzyme_support = support_batch[0]
# # #             substrate_support = support_batch[1]
# # #             kcat_support = support_batch[4]  # [K, 1]
# # #             cluster_support = support_batch[3]  # [K]
# # #
# # #             enzyme_query = query_batch[0]
# # #             substrate_query = query_batch[1]
# # #             kcat_query = query_batch[4]  # [N, 1]
# # #
# # #             # Stage 1：使用真实任务标签（簇编号）；Stage 2：由模型自动生成任务标签
# # #             if training_stage == 1:
# # #                 task_true_label = cluster_support[0].unsqueeze(0).float().to(enzyme_support.device)
# # #             else:
# # #                 task_true_label = None
# # #
# # #             # 1. 利用支持集计算任务特定权重向量
# # #             w = self.model.compute_task_weights(enzyme_support, substrate_support, kcat_support, task_true_label=task_true_label)
# # #             # 2. 对查询集进行预测
# # #             query_features = self.model.feature_extractor(enzyme_query, substrate_query)
# # #             y_pred = (query_features * w).sum(dim=1, keepdim=True)  # [N, 1]
# # #             loss = self.mse_loss(y_pred, kcat_query)
# # #             meta_loss += task_weight * loss
# # #
# # #         meta_loss = meta_loss / num_tasks
# # #         self.meta_optimizer.zero_grad()
# # #         meta_loss.backward()
# # #         self.meta_optimizer.step()
# # #         return meta_loss.item()
# # #
# # # # ----------------------------
# # # # 原 MAML 类（保留以供其他实验使用）
# # # # ----------------------------
# # # class MAML:
# # #     def __init__(self, model, inner_lr=0.01, meta_lr=0.001, n_inner_steps=10, lambda_metric=0.2):
# # #         self.model = model
# # #         self.inner_lr = inner_lr
# # #         self.meta_lr = meta_lr
# # #         self.n_inner_steps = n_inner_steps
# # #         self.lambda_metric = lambda_metric
# # #         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
# # #
# # #     def inner_loop(self, train_batch, criterion):
# # #         device = next(self.model.parameters()).device
# # #         adapted_model = deepcopy(self.model).to(device)
# # #         inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
# # #         for _ in range(self.n_inner_steps):
# # #             inner_optimizer.zero_grad()
# # #             output = adapted_model(train_batch[0], train_batch[1], train_batch[2], train_batch[3])
# # #             loss = criterion(output, train_batch[4])
# # #             loss.backward()
# # #             inner_optimizer.step()
# # #         return adapted_model
# # #
# # #     def meta_train_step(self, tasks, criterion):
# # #         meta_loss = 0.0
# # #         num_tasks = len(tasks)
# # #         for task in tasks:
# # #             if len(task) == 3:
# # #                 train_batch, val_batch, task_weight = task
# # #             else:
# # #                 train_batch, val_batch = task
# # #                 task_weight = 1.0
# # #             adapted_model = self.inner_loop(train_batch, criterion)
# # #             val_output = adapted_model(val_batch[0], val_batch[1], val_batch[2], val_batch[3])
# # #             loss_val = criterion(val_output, val_batch[4])
# # #             with torch.no_grad():
# # #                 train_features = adapted_model.feature_extractor(train_batch[0], train_batch[1])
# # #                 prototype = train_features.mean(dim=0, keepdim=True)
# # #             val_features = adapted_model.feature_extractor(val_batch[0], val_batch[1])
# # #             cos_sim = nn.functional.cosine_similarity(val_features, prototype.expand_as(val_features), dim=1)
# # #             metric_loss = 1 - cos_sim.mean()
# # #             total_loss = loss_val + self.lambda_metric * metric_loss
# # #             meta_loss += task_weight * total_loss
# # #         meta_loss = meta_loss / num_tasks
# # #         self.meta_optimizer.zero_grad()
# # #         meta_loss.backward()
# # #         self.meta_optimizer.step()
# # #         return meta_loss.item()
# #
# # # meta_learning.py
# # # meta_learning.py
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from copy import deepcopy
# #
# #
# # # ----------------------------
# # # FewShotRegression：针对少样本回归任务的元学习模块
# # # 该模块支持两阶段训练：
# # #   - Stage 1：使用真实任务标签（簇编号）训练特征提取器和权重生成器；
# # #   - Stage 2：冻结前两部分参数，更新 TaskLabelGenerator，并用其自动生成任务标签训练整个模型。
# # # ----------------------------
# # class FewShotRegression:
# #     def __init__(self, model, meta_lr=0.001):
# #         """
# #         model: EnhancedFewShotRegressionModel 实例，支持多温度输入，输出形状为 [N, T, 1]
# #         meta_lr: 元学习优化率
# #         """
# #         self.model = model
# #         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
# #         self.mse_loss = nn.MSELoss()
# #
# #     def meta_train_step(self, tasks, training_stage=1):
# #         """
# #         tasks: 列表，每个任务为 (support_batch, query_batch, task_weight)
# #         每个 batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
# #           其中 temp_embed 与 kcat_labels 的形状均为 [num_samples, T]
# #         training_stage: 1 表示第一阶段（使用真实任务标签），2 表示第二阶段（使用生成的任务标签）
# #         """
# #         meta_loss = 0.0
# #         num_tasks = len(tasks)
# #         # 根据训练阶段设置模型的标志
# #         if training_stage == 1:
# #             self.model.use_true_task_labels = True
# #         else:
# #             self.model.use_true_task_labels = False
# #
# #         for task in tasks:
# #             if len(task) == 3:
# #                 support_batch, query_batch, task_weight = task
# #             else:
# #                 support_batch, query_batch = task
# #                 task_weight = 1.0
# #
# #             # support_batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
# #             enzyme_support = support_batch[0]
# #             substrate_support = support_batch[1]
# #             temp_support = support_batch[2]  # [K, T]
# #             kcat_support = support_batch[4]  # [K, T]
# #             cluster_support = support_batch[3]  # [K,]
# #
# #             # query_batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
# #             enzyme_query = query_batch[0]
# #             substrate_query = query_batch[1]
# #             temp_query = query_batch[2]  # [N, T]
# #             kcat_query = query_batch[4]  # [N, T]
# #             # Stage 1：使用真实任务标签（簇编号）；Stage 2：由模型自动生成任务标签
# #             if training_stage == 1:
# #                 task_true_label = cluster_support[0].unsqueeze(0).float().to(enzyme_support.device)
# #             else:
# #                 task_true_label = None
# #
# #             # 1. 利用支持集计算任务特定权重向量
# #             w = self.model.compute_task_weights(enzyme_support, substrate_support, temp_support, kcat_support,
# #                                                 task_true_label=task_true_label)
# #             # 2. 对查询集进行预测，注意预测输出形状为 [N, T, 1]
# #             y_pred = self.model.predict(enzyme_query, substrate_query, temp_query, w)
# #             # 调整预测输出形状为 [N, T]以匹配目标
# #             y_pred = y_pred.squeeze(-1)
# #             loss = self.mse_loss(y_pred, kcat_query)
# #             meta_loss += task_weight * loss
# #
# #         meta_loss = meta_loss / num_tasks
# #         self.meta_optimizer.zero_grad()
# #         meta_loss.backward()
# #         self.meta_optimizer.step()
# #         return meta_loss.item()
# #
# #
# # # ----------------------------
# # # 原 MAML 类（保留以供其他实验使用）
# # # ----------------------------
# # class MAML:
# #     def __init__(self, model, inner_lr=0.01, meta_lr=0.001, n_inner_steps=10, lambda_metric=0.2):
# #         self.model = model
# #         self.inner_lr = inner_lr
# #         self.meta_lr = meta_lr
# #         self.n_inner_steps = n_inner_steps
# #         self.lambda_metric = lambda_metric
# #         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
# #
# #     def inner_loop(self, train_batch, criterion):
# #         device = next(self.model.parameters()).device
# #         adapted_model = deepcopy(self.model).to(device)
# #         inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
# #         for _ in range(self.n_inner_steps):
# #             inner_optimizer.zero_grad()
# #             # 假设 train_batch 格式为 (enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
# #             output = adapted_model(train_batch[0], train_batch[1], train_batch[2])
# #             loss = criterion(output, train_batch[4])
# #             loss.backward()
# #             inner_optimizer.step()
# #         return adapted_model
# #
# #     def meta_train_step(self, tasks, criterion):
# #         meta_loss = 0.0
# #         num_tasks = len(tasks)
# #         for task in tasks:
# #             if len(task) == 3:
# #                 train_batch, val_batch, task_weight = task
# #             else:
# #                 train_batch, val_batch = task
# #                 task_weight = 1.0
# #             adapted_model = self.inner_loop(train_batch, criterion)
# #             val_output = adapted_model(val_batch[0], val_batch[1], val_batch[2])
# #             loss_val = criterion(val_output, val_batch[4])
# #             with torch.no_grad():
# #                 train_features = adapted_model.feature_extractor(train_batch[0], train_batch[1])
# #                 prototype = train_features.mean(dim=0, keepdim=True)
# #             val_features = adapted_model.feature_extractor(val_batch[0], val_batch[1])
# #             cos_sim = nn.functional.cosine_similarity(val_features, prototype.expand_as(val_features), dim=1)
# #             metric_loss = 1 - cos_sim.mean()
# #             total_loss = loss_val + self.lambda_metric * metric_loss
# #             meta_loss += task_weight * total_loss
# #         meta_loss = meta_loss / num_tasks
# #         self.meta_optimizer.zero_grad()
# #         meta_loss.backward()
# #         self.meta_optimizer.step()
# #         return meta_loss.item()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from copy import deepcopy
#
# class FewShotRegression:
#     def __init__(self, model, meta_lr=0.001):
#         """
#         model: EnhancedFewShotRegressionModel 实例，包含 feature_extractor、weights_generator 和 task_label_generator
#         meta_lr: 元学习优化率
#         """
#         self.model = model
#         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
#         self.mse_loss = nn.MSELoss()
#
#     def meta_train_step(self, tasks, training_stage=1):
#         """
#         tasks: 列表，每个任务为 (support_batch, query_batch, task_weight)
#         每个 batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
#         training_stage: 1 表示第一阶段（使用真实任务标签），2 表示第二阶段（使用生成的任务标签）
#         """
#         meta_loss = 0.0
#         num_tasks = len(tasks)
#         if training_stage == 1:
#             self.model.use_true_task_labels = True
#         else:
#             self.model.use_true_task_labels = False
#
#         for task in tasks:
#             if len(task) == 3:
#                 support_batch, query_batch, task_weight = task
#             else:
#                 support_batch, query_batch = task
#                 task_weight = 1.0
#
#             # support_batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
#             enzyme_support = support_batch[0]
#             substrate_support = support_batch[1]
#             # 这里支持集的标签采用 kcat_labels（index 4），而非 temp_embed
#             kcat_support = support_batch[4]  # [K, 1]
#             cluster_support = support_batch[3]  # [K]
#
#             enzyme_query = query_batch[0]
#             substrate_query = query_batch[1]
#             kcat_query = query_batch[4]  # [N, 1]
#
#             if training_stage == 1:
#                 task_true_label = cluster_support[0].unsqueeze(0).float().to(enzyme_support.device)
#             else:
#                 task_true_label = None
#
#             # 修正调用：只传入 enzyme_support、substrate_support 和 kcat_support，同时通过 keyword 传入 task_true_label
#             w = self.model.compute_task_weights(enzyme_support, substrate_support, kcat_support, task_true_label=task_true_label)
#             # 对查询集进行预测
#             query_features = self.model.feature_extractor(enzyme_query, substrate_query)
#             y_pred = (query_features * w).sum(dim=1, keepdim=True)  # [N, 1]
#             loss = self.mse_loss(y_pred, kcat_query)
#             meta_loss += task_weight * loss
#
#         meta_loss = meta_loss / num_tasks
#         self.meta_optimizer.zero_grad()
#         meta_loss.backward()
#         self.meta_optimizer.step()
#         return meta_loss.item()
#
# class MAML:
#     def __init__(self, model, inner_lr=0.01, meta_lr=0.001, n_inner_steps=10, lambda_metric=0.2):
#         self.model = model
#         self.inner_lr = inner_lr
#         self.meta_lr = meta_lr
#         self.n_inner_steps = n_inner_steps
#         self.lambda_metric = lambda_metric
#         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
#
#     def inner_loop(self, train_batch, criterion):
#         device = next(self.model.parameters()).device
#         adapted_model = deepcopy(self.model).to(device)
#         inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
#         for _ in range(self.n_inner_steps):
#             inner_optimizer.zero_grad()
#             output = adapted_model(train_batch[0], train_batch[1], train_batch[2], train_batch[3])
#             loss = criterion(output, train_batch[4])
#             loss.backward()
#             inner_optimizer.step()
#         return adapted_model
#
#     def meta_train_step(self, tasks, criterion):
#         meta_loss = 0.0
#         num_tasks = len(tasks)
#         for task in tasks:
#             if len(task) == 3:
#                 train_batch, val_batch, task_weight = task
#             else:
#                 train_batch, val_batch = task
#                 task_weight = 1.0
#             adapted_model = self.inner_loop(train_batch, criterion)
#             val_output = adapted_model(val_batch[0], val_batch[1], val_batch[2], val_batch[3])
#             loss_val = criterion(val_output, val_batch[4])
#             with torch.no_grad():
#                 train_features = adapted_model.feature_extractor(train_batch[0], train_batch[1])
#                 prototype = train_features.mean(dim=0, keepdim=True)
#             val_features = adapted_model.feature_extractor(val_batch[0], val_batch[1])
#             cos_sim = nn.functional.cosine_similarity(val_features, prototype.expand_as(val_features), dim=1)
#             metric_loss = 1 - cos_sim.mean()
#             total_loss = loss_val + self.lambda_metric * metric_loss
#             meta_loss += task_weight * total_loss
#         meta_loss = meta_loss / num_tasks
#         self.meta_optimizer.zero_grad()
#         meta_loss.backward()
#         self.meta_optimizer.step()
#         return meta_loss.item()
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class FewShotRegression:
    def __init__(self, model, meta_lr=0.001):
        """
        model: EnhancedFewShotRegressionModel 实例，包含 feature_extractor、weights_generator 和 task_label_generator
        meta_lr: 元学习优化率
        """
        self.model = model
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.mse_loss = nn.MSELoss()

    def meta_train_step(self, tasks, training_stage=1):
        """
        tasks: 列表，每个任务为 (support_batch, query_batch, task_weight)
        每个 batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
        training_stage: 1 表示第一阶段（使用真实任务标签），2 表示第二阶段（使用生成的任务标签）
        """
        meta_loss = 0.0
        num_tasks = len(tasks)
        if training_stage == 1:
            self.model.use_true_task_labels = True
        else:
            self.model.use_true_task_labels = False

        for task in tasks:
            if len(task) == 3:
                support_batch, query_batch, task_weight = task
            else:
                support_batch, query_batch = task
                task_weight = 1.0

            # support_batch 格式：(enzyme_feat, substrate_feat, temp_embed, cluster_ids, kcat_labels)
            enzyme_support = support_batch[0]
            substrate_support = support_batch[1]
            # 这里支持集的标签采用 kcat_labels（index 4），而非 temp_embed
            kcat_support = support_batch[4]  # [K, 1]
            cluster_support = support_batch[3]  # [K]

            enzyme_query = query_batch[0]
            substrate_query = query_batch[1]
            kcat_query = query_batch[4]  # [N, 1]

            if training_stage == 1:
                task_true_label = cluster_support[0].unsqueeze(0).float().to(enzyme_support.device)
            else:
                task_true_label = None

            # 修正调用：只传入 enzyme_support、substrate_support 和 kcat_support，同时通过 keyword 传入 task_true_label
            w = self.model.compute_task_weights(enzyme_support, substrate_support, kcat_support, task_true_label=task_true_label)
            # 对查询集进行预测
            query_features = self.model.feature_extractor(enzyme_query, substrate_query)
            y_pred = (query_features * w).sum(dim=1, keepdim=True)  # [N, 1]
            loss = self.mse_loss(y_pred, kcat_query)
            meta_loss += task_weight * loss

        meta_loss = meta_loss / num_tasks
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()

class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, n_inner_steps=10, lambda_metric=0.2):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
        self.lambda_metric = lambda_metric
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)

    def inner_loop(self, train_batch, criterion):
        device = next(self.model.parameters()).device
        adapted_model = deepcopy(self.model).to(device)
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        for _ in range(self.n_inner_steps):
            inner_optimizer.zero_grad()
            output = adapted_model(train_batch[0], train_batch[1], train_batch[2], train_batch[3])
            loss = criterion(output, train_batch[4])
            loss.backward()
            inner_optimizer.step()
        return adapted_model

    def meta_train_step(self, tasks, criterion):
        meta_loss = 0.0
        num_tasks = len(tasks)
        for task in tasks:
            if len(task) == 3:
                train_batch, val_batch, task_weight = task
            else:
                train_batch, val_batch = task
                task_weight = 1.0
            adapted_model = self.inner_loop(train_batch, criterion)
            val_output = adapted_model(val_batch[0], val_batch[1], val_batch[2], val_batch[3])
            loss_val = criterion(val_output, val_batch[4])
            with torch.no_grad():
                train_features = adapted_model.feature_extractor(train_batch[0], train_batch[1])
                prototype = train_features.mean(dim=0, keepdim=True)
            val_features = adapted_model.feature_extractor(val_batch[0], val_batch[1])
            cos_sim = nn.functional.cosine_similarity(val_features, prototype.expand_as(val_features), dim=1)
            metric_loss = 1 - cos_sim.mean()
            total_loss = loss_val + self.lambda_metric * metric_loss
            meta_loss += task_weight * total_loss
        meta_loss = meta_loss / num_tasks
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()
