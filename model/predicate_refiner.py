"""
谓词微调分类器 - 基于原型学习的谓词精调器

关键改进：
1. 使用VR特征（独立于GGNN，避免影响主分支）
2. 原型用NODE_EMBEDDING预训练嵌入初始化（有语义先验）
3. 使用Triplet Loss替代多样性损失（同时推+拉）
4. 聚焦优化可改进的谓词（8个，错误率40-70%）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle

from config import VG_SGG_DICT_FN, NODE_EMBEDDING

# 可优化的目标谓词列表（错误率40-70%，有明显视觉特征，样本数>100）
TARGET_PREDICATES = [
    'behind',         # 47.3%错误率，空间关系清晰
    'looking at',     # 48.4%错误率，有方向性
    'sitting on',     # 53.6%错误率，有姿态特征
    'holding',        # 62.6%错误率，视觉明显（手拿着）
    'under',          # 61.1%错误率，空间关系清晰
    'wearing',        # 73.9%错误率，视觉明显（衣服）
    'sitting on',
    'in front of'
]


class PredicateRefiner(nn.Module):
    """
    基于原型学习的谓词精调器

    特点：
    - 使用预训练的谓词嵌入初始化原型（从NODE_EMBEDDING加载）
    - 使用Triplet Loss训练（拉近正样本，推开负样本）
    - 接收VR特征作为输入（独立于主分支GGNN）
    - 只优化8个可改进的谓词
    """
    def __init__(self, feature_dim=4096, triplet_margin=1.0):
        super(PredicateRefiner, self).__init__()

        self.feature_dim = feature_dim
        self.triplet_margin = triplet_margin

        # 加载谓词字典
        self.predicate_dict_loader = PredicateDictLoader()

        # 目标谓词列表（用于损失计算时的聚焦优化）
        self.target_predicates = TARGET_PREDICATES
        self.num_target_predicates = len(self.target_predicates)

        # 加载NODE_EMBEDDING，获取预训练的谓词嵌入
        try:
            with open(NODE_EMBEDDING, 'rb') as f:
                emb_ent, emb_pred = pickle.load(f)
            # emb_pred形状: [51, embedding_dim]
            self.pred_embedding_dim = emb_pred.shape[1]
        except Exception as e:
            print(f"Warning: Failed to load NODE_EMBEDDING: {e}")
            print("Falling back to random initialization")
            self.pred_embedding_dim = 300
            emb_pred = None

        # 特征投影层：将VR特征投影到谓词嵌入空间
        self.feature_projector = nn.Linear(feature_dim, self.pred_embedding_dim)

        # 原型嵌入：使用预训练的谓词嵌入初始化（只针对8个目标谓词）
        if emb_pred is not None:
            # 确保 emb_pred 是 tensor
            if isinstance(emb_pred, np.ndarray):
                emb_pred = torch.from_numpy(emb_pred).float()
            elif not isinstance(emb_pred, torch.Tensor):
                emb_pred = torch.tensor(emb_pred, dtype=torch.float32)
            else:
                emb_pred = emb_pred.float()

            # 从51个谓词中提取目标谓词的嵌入
            target_embeddings = []
            for pred_name in self.target_predicates:
                pred_idx = self.predicate_dict_loader.predicate_to_idx.get(pred_name)
                if pred_idx is not None and pred_idx < emb_pred.shape[0]:
                    target_embeddings.append(emb_pred[pred_idx])
                else:
                    # 如果找不到，用随机初始化
                    target_embeddings.append(torch.randn(self.pred_embedding_dim) * 0.1)

            # 堆叠成原型矩阵（8个原型）
            prototype_init = torch.stack(target_embeddings)
            self.prototype_emb = nn.Parameter(prototype_init, requires_grad=True)
        else:
            # 回退到随机初始化（8个原型）
            self.prototype_emb = nn.Parameter(
                torch.randn(self.num_target_predicates, self.pred_embedding_dim) * 0.1,
                requires_grad=True
            )
            nn.init.orthogonal_(self.prototype_emb)

        # 距离缩放参数（用于将距离转换为logits）
        self.negative_scale = nn.Parameter(torch.ones(1) * 15)
        self.shift = nn.Parameter(torch.ones(1) * 15)

        # 预计算标签映射缓存，加速训练
        self.vg_idx_to_internal_cache = {}
        for vg_idx, pred_name in self.predicate_dict_loader.idx_to_predicate.items():
            if pred_name in self.target_predicates:
                self.vg_idx_to_internal_cache[int(vg_idx)] = self.target_predicates.index(pred_name)

        self._init_parameters()

    def _init_parameters(self):
        """初始化网络参数"""
        nn.init.xavier_uniform_(self.feature_projector.weight)
        nn.init.constant_(self.feature_projector.bias, 0)

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

        # 创建掩码：标记哪些关系是我们的目标谓词
        if target_labels is not None:
            # 训练时：使用 torch.isin 进行向量化成员检查
            target_indices_tensor = torch.tensor(target_indices, device=device)
            valid_mask = torch.isin(target_labels, target_indices_tensor)
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
        projected_features = self.feature_projector(valid_features)  # [num_valid, embedding_dim]

        # 获取原型参数
        prototypes = self.prototype_emb  # [num_target_predicates, embedding_dim]

        # 标准化特征和原型（L2归一化）
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        projected_features_norm = F.normalize(projected_features, p=2, dim=1)

        # 计算距离矩阵
        rel_rep_expand = projected_features_norm.unsqueeze(1).expand(-1, self.num_target_predicates, -1)
        prototype_expand = prototypes_norm.unsqueeze(0).expand(projected_features_norm.shape[0], -1, -1)

        # 计算欧氏距离
        distance_set = (rel_rep_expand - prototype_expand).norm(dim=2)  # [num_valid, num_target_predicates]

        # 转换为相似度分数（距离越小，分数越高）
        rel_scores = -self.negative_scale * distance_set + self.shift

        # 转换为概率分布
        rel_probs = F.softmax(rel_scores, dim=1)  # [num_valid, num_target_predicates]

        # 计算置信度（基于最大概率）
        confidences = rel_probs.max(dim=1)[0]  # [num_valid]

        # 创建完整的输出张量
        output_probs = torch.zeros(num_relations, self.num_target_predicates, device=device)
        output_probs[valid_mask] = rel_probs

        output_confidences = torch.zeros(num_relations, device=device)
        output_confidences[valid_mask] = confidences

        if self.training and target_labels is not None:
            # 训练时：需要转换标签并计算损失
            valid_target_labels = target_labels[valid_mask]

            # 向量化标签转换
            if not hasattr(self, 'label_mapping_tensor') or self.label_mapping_tensor.device != device:
                max_idx = max(self.vg_idx_to_internal_cache.keys()) if self.vg_idx_to_internal_cache else -1
                if max_idx < 0:
                    return output_probs, {}, valid_mask

                mapping = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
                for vg_idx, internal_idx in self.vg_idx_to_internal_cache.items():
                    mapping[vg_idx] = internal_idx
                self.label_mapping_tensor = mapping

            internal_labels = self.label_mapping_tensor[valid_target_labels]

            # 过滤无效标签
            valid_internal_mask = internal_labels >= 0
            if not valid_internal_mask.all():
                valid_internal_indices = valid_internal_mask.nonzero().squeeze()
                projected_features = projected_features[valid_internal_indices]
                internal_labels = internal_labels[valid_internal_indices]
                prototypes_norm = prototypes_norm[valid_internal_indices] if valid_internal_indices.shape[0] < prototypes_norm.shape[0] else prototypes_norm

            # 计算Triplet Loss
            loss_dict = self._compute_triplet_loss(
                projected_features, prototypes_norm, internal_labels, distance_set
            )
            return output_probs, loss_dict, valid_mask
        else:
            return output_probs, output_confidences, valid_mask

    def _compute_triplet_loss(self, features, prototypes, target_labels, distance_set):
        """
        计算Triplet Loss：拉近正样本，推开负样本
        """
        loss_dict = {}
        device = features.device

        num_samples = target_labels.shape[0]
        if num_samples == 0:
            return loss_dict

        # 正样本距离
        pos_dist = distance_set[torch.arange(num_samples), target_labels]

        # 负样本距离（除了正样本之外的所有距离）
        # 创建掩码，将正样本位置设为inf
        mask = torch.ones_like(distance_set)
        mask[torch.arange(num_samples), target_labels] = float('inf')

        # 获取负样本距离，并取最小的5个
        neg_distances = mask * distance_set
        neg_dist, _ = torch.topk(neg_distances, k=min(5, distance_set.shape[1] - 1), dim=1, largest=False)
        neg_dist = neg_dist.mean(dim=1)

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        triplet_loss = torch.clamp(pos_dist - neg_dist + self.triplet_margin, min=0.0).mean()

        loss_dict['triplet_loss'] = triplet_loss

        # 添加原型正交性损失作为辅助
        if prototypes.shape[0] > 1:
            proto_sim = torch.matmul(prototypes, prototypes.t())
            mask = torch.eye(proto_sim.size(0), device=device).bool()
            proto_sim_off_diag = proto_sim[~mask]
            ortho_loss = (proto_sim_off_diag ** 2).mean()
            loss_dict['orthogonal_loss'] = 0.1 * ortho_loss

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

    def apply_dpl_and_fuse(self, original_input, vr_features, target_labels=None,
                          enable_fusion=True):
        """
        应用 DPL 优化并融合到原始预测中

        Args:
            original_input: 原始关系预测 logits 或概率 [num_relations, num_rel_cls]
            vr_features: VR视觉特征 [num_relations, feature_dim]
            target_labels: 训练时的真实谓词标签 [num_relations] (可选)
            enable_fusion: 是否启用DPL融合

        Returns:
            fused_probs: DPL优化融合后的关系预测 [num_relations, num_rel_cls]
            loss_dict: DPL损失字典 (训练时返回)
        """
        # 获取目标谓词索引
        target_indices = self.get_target_predicate_indices()

        if not target_indices:
            if self.training and target_labels is not None:
                return original_input, {}
            else:
                return original_input

        # 使用VR特征进行 DPL 优化
        if self.training and target_labels is not None:
            # 训练模式：前向传播并计算损失
            dpl_rel_probs, loss_dict, valid_mask = self(vr_features, target_labels=target_labels)
            confidences = None  # 训练时不需要置信度
        else:
            # 推理模式：禁用梯度（不改变训练模式状态）
            with torch.no_grad():
                dpl_rel_probs, confidences, _ = self(vr_features)
            loss_dict = None

        # 如果禁用融合，直接返回原始预测（但仍然计算DPL损失用于训练）
        if not enable_fusion:
            if self.training and target_labels is not None:
                return original_input, loss_dict
            else:
                return original_input

        # 转换 DPL 预测到全局概率空间
        dpl_global_probs = torch.zeros_like(original_input)
        for i, global_idx in enumerate(target_indices):
            if global_idx < original_input.shape[1]:
                dpl_global_probs[:, global_idx] = dpl_rel_probs[:, i]

        # 固定融合策略：主分支90% + DPL 10%
        alpha = 0.9  # 主分支权重
        beta = 0.1   # DPL权重

        # 计算原始分布中目标谓词和非目标谓词的概率和
        target_indices_tensor = torch.tensor(target_indices, device=original_input.device)
        original_target_sum = original_input[:, target_indices_tensor].sum(dim=1, keepdim=True)  # [N, 1]
        original_non_target_sum = 1 - original_target_sum  # [N, 1]

        # 对目标谓词进行融合
        fused_probs = original_input.clone()
        for i, target_idx in enumerate(target_indices):
            fused_probs[:, target_idx] = (
                alpha * original_input[:, target_idx] +
                beta * dpl_global_probs[:, target_idx]
            )

        # 计算融合后目标谓词的新概率和
        fused_target_sum = fused_probs[:, target_indices_tensor].sum(dim=1, keepdim=True)  # [N, 1]

        # 计算缩放因子：保持非目标谓词的相对概率，但调整总和
        # 新的非目标和 = 1 - 融合后的目标和
        new_non_target_sum = 1 - fused_target_sum  # [N, 1]

        # 对每个非目标谓词按比例缩放
        for idx in range(original_input.shape[1]):
            if idx not in target_indices:
                # 保持相对概率，按比例缩放
                fused_probs[:, idx] = original_input[:, idx] * new_non_target_sum.squeeze() / (original_non_target_sum.squeeze() + 1e-8)

        # 确保概率和为1（最后归一化，防止浮点误差）
        prob_sums = fused_probs.sum(dim=1, keepdim=True)
        fused_probs = fused_probs / prob_sums

        # 保持概率空间，避免双重 log 导致 NaN
        # 外层的 F_nll_loss 会自动处理 log
        if self.training and target_labels is not None:
            return fused_probs, loss_dict
        else:
            return fused_probs


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
