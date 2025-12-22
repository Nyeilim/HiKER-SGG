"""
谓词微调分类器 - 基于 DPL (Semantic Diversity-aware Prototype-based Learning) 的实现
针对高错误率谓词进行专门优化，使用高斯分布参数化和多样性感知损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.cuda import current_device

from config import VG_SGG_DICT_FN

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')

# 需要优化的目标谓词列表
TARGET_PREDICATES = [
    'flying in',
    'lying on',
    'walking in',
    'mounted on',
    'part of',
    'playing',
    'on',
    'says',
    'above',
    'across',
]


class PredicateRefiner(nn.Module):
    """
    基于 DPL (Semantic Diversity-aware Prototype-based Learning) 的谓词精调器
    针对指定谓词进行专门优化，使用高斯分布参数化和多样性感知损失
    直接输出概率分布用于融合
    """
    def __init__(self, feature_dim=1024, prototype_dim=128, avg_sample_size=20,
                 alpha=10.0, radius=1.0):
        super(PredicateRefiner, self).__init__()

        self.feature_dim = feature_dim
        self.prototype_dim = prototype_dim
        self.avg_sample_size = avg_sample_size
        self.alpha = alpha
        self.radius = radius

        # 加载谓词字典
        self.predicate_dict_loader = PredicateDictLoader()

        # 目标谓词列表
        self.target_predicates = TARGET_PREDICATES
        self.num_target_predicates = len(self.target_predicates)

        # 特征投影层：将输入特征投影到原型空间
        self.feature_projector = nn.Linear(feature_dim, prototype_dim)

        # 原型嵌入：每个目标谓词的原型向量
        self.prototype_emb = nn.Parameter(
            torch.randn(self.num_target_predicates, prototype_dim) * 0.1,
            requires_grad=True
        )
        nn.init.orthogonal_(self.prototype_emb)

        # 高斯分布参数化：将原型映射到高斯分布的均值和标准差
        self.gaussian_emb = nn.Linear(prototype_dim, prototype_dim * 2)

        # 距离缩放参数
        self.negative_scale = nn.Parameter(torch.ones(1) * 15)
        self.shift = nn.Parameter(torch.ones(1) * 15)

        # 固定采样大小
        self.sample_size_array = torch.ones(self.num_target_predicates, dtype=torch.int32) * avg_sample_size

        # 损失计算器
        self.loss_calculator = DPLossCalculator(alpha=alpha, radius=radius)

        # 初始化参数
        self._init_parameters()

    
    def _init_parameters(self):
        """初始化网络参数"""
        nn.init.xavier_uniform_(self.feature_projector.weight)
        nn.init.constant_(self.feature_projector.bias, 0)
        nn.init.xavier_uniform_(self.gaussian_emb.weight)
        nn.init.constant_(self.gaussian_emb.bias, 0)

    def forward(self, features, target_labels=None):
        """
        前向传播

        Args:
            features: 关系特征 [num_relations, feature_dim]
            target_labels: 训练时的真实标签 [num_relations] (可选，使用VG真实索引)

        Returns:
            训练时: (rel_probs, loss_dict, valid_mask)
            推理时: (rel_probs, confidences, valid_mask)
        """
        num_relations = features.shape[0]
        device = features.device

        # 获取目标谓词的真实索引
        target_indices = self.get_target_predicate_indices()
        target_idx_set = set(target_indices)

        # 创建掩码：标记哪些关系是我们的目标谓词
        if target_labels is not None:
            # 训练时：根据target_labels过滤
            valid_mask = torch.tensor([label.item() in target_idx_set for label in target_labels],
                                    device=device, dtype=torch.bool)
        else:
            # 推理时：处理所有关系
            valid_mask = torch.ones(num_relations, device=device, dtype=torch.bool)

        if not valid_mask.any():
            # 如果没有目标谓词，返回空结果
            empty_probs = torch.zeros(num_relations, self.num_target_predicates, device=device)
            empty_confidences = torch.zeros(num_relations, device=device)
            if self.training and target_labels is not None:
                return empty_probs, {}, valid_mask
            else:
                return empty_probs, empty_confidences, valid_mask

        # 只对有效关系进行处理
        valid_features = features[valid_mask]  # [num_valid, feature_dim]

        # 投影特征到原型空间
        projected_features = self.feature_projector(valid_features)  # [num_valid, prototype_dim]

        # 获取原型参数
        prototypes = self.prototype_emb  # [num_target_predicates, prototype_dim]

        # 计算高斯分布参数
        gaussian_params = self.gaussian_emb(prototypes)  # [num_target_predicates, prototype_dim * 2]
        mu, logsigma = torch.split(gaussian_params, self.prototype_dim, dim=1)  # 各 [num_target_predicates, prototype_dim]

        # 标准化原型
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        projected_features_norm = F.normalize(projected_features, p=2, dim=1)

        # 计算距离矩阵
        rel_rep_expand = projected_features_norm.unsqueeze(1).expand(-1, self.num_target_predicates, -1)
        prototype_expand = prototypes_norm.unsqueeze(0).expand(projected_features_norm.shape[0], -1, -1)

        # 计算加权距离 (考虑高斯分布的不确定性)
        distance_set = (rel_rep_expand - prototype_expand).norm(dim=2)

        # 使用负缩放和偏移转换为相似度分数
        rel_scores = -self.negative_scale * distance_set + self.shift

        # 直接转换为概率分布输出
        valid_rel_probs = F.softmax(rel_scores, dim=1)  # [num_valid, num_target_predicates], 范围: 0~1

        # 计算置信度（基于最大概率）
        valid_confidences = valid_rel_probs.max(dim=1)[0]

        # 创建完整的输出张量
        rel_probs = torch.zeros(num_relations, self.num_target_predicates, device=device)
        rel_probs[valid_mask] = valid_rel_probs

        if self.training and target_labels is not None:
            # 训练时：需要转换标签
            valid_target_labels = target_labels[valid_mask]
            internal_labels = []
            for vg_idx in valid_target_labels:
                pred_name = self.predicate_dict_loader.idx_to_predicate.get(int(vg_idx), None)
                if pred_name in self.target_predicates:
                    internal_labels.append(self.target_predicates.index(pred_name))
                else:
                    internal_labels.append(0)  # 默认值，虽然理论上不应该发生

            internal_labels = torch.tensor(internal_labels, device=device)

            # 计算损失
            loss_dict = self._compute_dpl_losses(
                projected_features, prototypes_norm, logsigma, internal_labels
            )
            return rel_probs, loss_dict, valid_mask
        else:
            # 推理时：返回概率和置信度
            confidences = torch.zeros(num_relations, device=device)
            confidences[valid_mask] = valid_confidences
            return rel_probs, confidences, valid_mask

    def _compute_dpl_losses(self, features, prototypes, logsigma, target_labels):
        """计算DPL的各个损失项"""
        loss_dict = {}

        # 1. 原型正交性损失
        proto_sim = torch.matmul(prototypes, prototypes.t())
        ortho_loss = self.loss_calculator.get_orthogonal_loss(proto_sim)
        loss_dict['orthogonal_loss'] = ortho_loss

        # 2. 多样性感知损失
        diversity_loss = self.loss_calculator.get_diversity_loss(
            features, prototypes, logsigma, self.sample_size_array, target_labels
        )
        loss_dict['diversity_loss'] = diversity_loss

        return loss_dict

    def get_target_predicate_indices(self):
        """
        获取目标谓词在VG数据集中的真实索引
        """
        # 直接使用谓词字典中的真实索引
        target_indices = []
        for pred_name in self.target_predicates:
            if pred_name in self.predicate_dict_loader.predicate_to_idx:
                real_idx = self.predicate_dict_loader.predicate_to_idx[pred_name]
                target_indices.append(real_idx)

        return target_indices

    def apply_dpl_and_fuse(self, original_input, enhanced_features, target_labels=None):
        """
        应用 DPL 优化并融合到原始预测中

        Args:
            original_input: 原始关系预测 logits 或概率 [num_relations, num_rel_cls]
            enhanced_features: GGNN语义增强后的特征 [num_relations, feature_dim]
            target_labels: 训练时的真实谓词标签 [num_relations] (可选)

        Returns:
            fused_logits: DPL优化融合后的关系预测 [num_relations, num_rel_cls]
            loss_dict: DPL损失字典 (训练时返回)
        """
        # 获取目标谓词索引
        target_indices = self.get_target_predicate_indices()

        if not target_indices:
            return original_input

        # 使用增强后的关系特征进行 DPL 优化
        if self.training and target_labels is not None:
            # 训练模式：前向传播并计算损失
            dpl_rel_probs, loss_dict, valid_mask = self(enhanced_features, target_labels=target_labels)
            confidences = None  # 训练时不需要置信度
        else:
            # 推理模式：禁用梯度
            self.eval()
            with torch.no_grad():
                dpl_rel_probs, confidences, _ = self(enhanced_features)
            loss_dict = None

        # 转换 DPL 预测到全局概率空间
        dpl_global_probs = torch.zeros_like(original_input)
        for i, global_idx in enumerate(target_indices):
            if global_idx < original_input.shape[1]:
                dpl_global_probs[:, global_idx] = dpl_rel_probs[:, i]

        # 直接使用概率空间（输入已经是概率）
        original_probs = original_input

        # 每个关系单独进行概率空间加权
        fused_probs = original_probs.clone()

        # 使用每个关系的单独置信度（训练和推理统一）
        for i in range(original_probs.shape[0]):
            if self.training:
                # 训练时：由于没有置信度，使用默认的保守策略
                alpha = 0.9  # 主要保持原样，给DPL轻微的学习机会
            else:
                # 推理时：基于置信度决定是否使用DPL
                confidence = confidences[i].item()

                # 置信度太低，直接忽略DPL结果
                if confidence < 0.2:
                    continue  # 跳过DPL融合，保持原始预测

                # 置信度足够高，进行融合（只做上界限制）
                confidence = min(confidence, 0.9)

                # 动态融合权重（基于该关系的置信度）
                if confidence > 0.8:
                    alpha = 0.3  # 更多依赖 DPL
                elif confidence > 0.6:
                    alpha = 0.5  # 平衡融合
                elif confidence > 0.4:
                    alpha = 0.7  # 主要保持原样
                else:
                    alpha = 0.9  # 低置信度但可用，主要保持原样

            # 对该关系进行概率空间融合
            for target_idx in target_indices:
                fused_probs[i, target_idx] = (
                    alpha * original_probs[i, target_idx] +
                    (1 - alpha) * dpl_global_probs[i, target_idx]
                )

        # 重新归一化每一行，确保概率和为1
        fused_probs = fused_probs / fused_probs.sum(dim=1, keepdim=True)

        # 转换回 logits 空间
        fused_logits = torch.log(fused_probs + 1e-8)

        if self.training and target_labels is not None:
            return fused_logits, loss_dict
        else:
            return fused_logits


class PredicateDictLoader:
    """谓词字典加载器"""
    def __init__(self):
        self.idx_to_predicate = {}
        self.predicate_to_idx = {}
        self.is_loaded = False
        self._load_predicate_dict()

    def _load_predicate_dict(self):
        """从 VG_SGG_DICT_FN 加载谓词字典"""
        with open(str(VG_SGG_DICT_FN), 'r') as f:
            data = json.load(f)

        self.idx_to_predicate = {int(k): v for k, v in data['idx_to_predicate'].items()}
        self.predicate_to_idx = data['predicate_to_idx']
        self.is_loaded = True


class DPLossCalculator:
    """DPL 特殊损失函数计算器"""

    def __init__(self, alpha=10.0, radius=1.0):
        self.alpha = alpha
        self.radius = radius

    def get_orthogonal_loss(self, proto_sim):
        """
        计算原型正交性损失
        Args:
            proto_sim: 原型相似度矩阵 [num_classes, num_classes]
        """
        eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
        loss_orth = torch.abs(proto_sim[eye_sim == 1]).mean()
        return loss_orth

    def get_diversity_loss(self, features, prototypes, logsigma, sample_size_array, target_labels):
        """
        计算多样性感知损失

        Args:
            features: 关系特征 [num_relations, feature_dim]
            prototypes: 原型向量 [num_classes, feature_dim]
            logsigma: 高斯分布对数标准差 [num_classes, feature_dim]
            sample_size_array: 每个类的采样数量 [num_classes]
            target_labels: 目标标签 [num_relations]
        """
        num_relations = features.shape[0]
        device = features.device

        total_loss = 0.0
        count = 0

        for i in range(num_relations):
            target_idx = target_labels[i].item()
            target_prototype = prototypes[target_idx]
            target_logsigma = logsigma[target_idx]
            num_samples = sample_size_array[target_idx]

            # 从目标原型的高斯分布中采样
            eps = torch.randn(num_samples, prototypes.shape[1], device=device)
            samples = eps * torch.exp(target_logsigma) + target_prototype

            # 计算当前特征到采样点的距离
            distances = torch.norm(samples - features[i].unsqueeze(0), dim=1)

            # 计算多样性损失
            min_distance = distances.min()
            if min_distance > self.radius:
                loss = (min_distance - self.radius) ** 2
                total_loss += loss
                count += 1

        if count > 0:
            return self.alpha * (total_loss / count)
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)