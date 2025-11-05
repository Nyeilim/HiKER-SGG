"""
谓词微调分类器 - 基于 DPL (Semantic Diversity-aware Prototype-based Learning) 的实现
针对高错误率谓词进行专门优化，使用高斯分布参数化和多样性感知损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
from torch.cuda import current_device

from config import VG_SGG_DICT_FN
from torch.cuda import current_device

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')

# 基于 requires.md 的前10个高错误率谓词
TOP_ERROR_PREDICATES = {
    'flying in': {'error_rate': 1.0000, 'sample_count': 25},
    'lying on': {'error_rate': 1.0000, 'sample_count': 170},
    'walking in': {'error_rate': 0.9381, 'sample_count': 113},
    'mounted on': {'error_rate': 0.9290, 'sample_count': 169},
    'part of': {'error_rate': 0.9085, 'sample_count': 142},
    'playing': {'error_rate': 0.8966, 'sample_count': 29},
    'on': {'error_rate': 0.8836, 'sample_count': 63023},
    'says': {'error_rate': 0.8333, 'sample_count': 12},
    'above': {'error_rate': 0.8300, 'sample_count': 2388},
    'across': {'error_rate': 0.8171, 'sample_count': 82},
}


class PredicateRefiner(nn.Module):
    """
    基于 DPL (Semantic Diversity-aware Prototype-based Learning) 的谓词精调器
    针对高错误率谓词进行专门优化，使用高斯分布参数化和多样性感知损失
    """
    def __init__(self, feature_dim=4096, prototype_dim=128, avg_sample_size=20,
                 alpha=10.0, radius=1.0, freq_based_sample=True,
                 temperature=2.0, logit_range=(-10, 10)):
        super(PredicateRefiner, self).__init__()

        self.feature_dim = feature_dim
        self.prototype_dim = prototype_dim
        self.avg_sample_size = avg_sample_size
        self.alpha = alpha
        self.radius = radius
        self.temperature = temperature  # 温度参数，控制输出尖锐度
        self.logit_min, self.logit_max = logit_range  # logits范围限制

        # 加载谓词字典
        self.predicate_dict_loader = PredicateDictLoader()

        # 目标谓词列表
        self.target_predicates = list(TOP_ERROR_PREDICATES.keys())
        self.num_target_predicates = len(self.target_predicates)

        # 创建谓词到索引的映射
        self.target_predicate_to_idx = {pred: idx for idx, pred in enumerate(self.target_predicates)}

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

        # 自适应采样大小
        if freq_based_sample:
            self.sample_size_array = self._compute_adaptive_sample_size()
        else:
            self.sample_size_array = torch.ones(self.num_target_predicates, dtype=torch.int32) * avg_sample_size

        # 损失计算器
        self.loss_calculator = DPLossCalculator(alpha=alpha, radius=radius)

        # 初始化参数
        self._init_parameters()

    def _compute_adaptive_sample_size(self):
        """基于错误率计算自适应采样大小"""
        # 使用错误率作为"频率"的代理，错误率越高说明样本越少
        error_rates = [TOP_ERROR_PREDICATES[pred]['error_rate'] for pred in self.target_predicates]
        error_rates = np.array(error_rates)

        # 避免log(0)
        error_rates = np.maximum(error_rates, 0.01)

        # 基于错误率的对数计算采样大小
        log_error = np.log(error_rates)
        mean_log_error = np.mean(log_error)

        sample_sizes = np.round(log_error / mean_log_error * self.avg_sample_size).astype(int)
        sample_sizes = np.maximum(sample_sizes, 1)  # 确保至少采样1个

        return torch.tensor(sample_sizes, dtype=torch.int32)

    def _init_parameters(self):
        """初始化网络参数"""
        nn.init.xavier_uniform_(self.feature_projector.weight)
        nn.init.constant_(self.feature_projector.bias, 0)
        nn.init.xavier_uniform_(self.gaussian_emb.weight)
        nn.init.constant_(self.gaussian_emb.bias, 0)

    def forward(self, features, subject_indices=None, object_indices=None, predicate_indices=None, target_labels=None):
        """
        前向传播

        Args:
            features: 关系特征 [num_relations, feature_dim]
            subject_indices: 主语索引 [num_relations] (保留接口兼容性)
            object_indices: 宾语索引 [num_relations] (保留接口兼容性)
            predicate_indices: 谓词索引 [num_relations] (保留接口兼容性)
            target_labels: 训练时的真实标签 [num_relations] (可选)

        Returns:
            训练时: (rel_dists, loss_dict)
            推理时: (rel_dists, confidences)
        """
        num_relations = features.shape[0]
        device = features.device

        # 投影特征到原型空间
        projected_features = self.feature_projector(features)  # [num_relations, prototype_dim]

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
        prototype_expand = prototypes_norm.unsqueeze(0).expand(num_relations, -1, -1)

        # 计算加权距离 (考虑高斯分布的不确定性)
        distance_set = (rel_rep_expand - prototype_expand).norm(dim=2)

        # 使用负缩放和偏移转换为相似度分数
        if self.training:
            # 训练时使用标准距离
            rel_dists = -self.negative_scale * distance_set + self.shift
        else:
            # 推理时使用归一化距离
            nd = distance_set / logsigma.exp().max(dim=1)[0].unsqueeze(0)
            ndn = (nd.t() / nd.max(dim=1)[0] * distance_set.max(dim=1)[0]).t()
            rel_dists = -self.negative_scale * ndn + self.shift

        # 关键：将DPL输出转换为概率，然后再转换为对齐的logits
        dpl_probabilities = F.softmax(rel_dists, dim=1)  # [num_relations, 10], 范围: 0~1

        # 将概率转换回logits空间，模拟原始分类器的数量级
        # 使用温度参数控制输出的"尖锐度"
        aligned_logits = torch.log(dpl_probabilities + 1e-8) / self.temperature

        # 缩放到合理的logits范围，与原始分类器对齐
        aligned_logits = torch.clamp(aligned_logits, min=self.logit_min, max=self.logit_max)

        # 计算置信度（基于最大概率）
        confidences = dpl_probabilities.max(dim=1)[0]

        if self.training and target_labels is not None:
            # 训练时：返回原始距离用于损失计算，但对齐的logits用于融合
            loss_dict = self._compute_dpl_losses(
                projected_features, prototypes_norm, logsigma, target_labels
            )
            return aligned_logits, loss_dict
        else:
            # 推理时：返回对齐的logits和置信度
            return aligned_logits, confidences

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

    def get_target_predicate_indices(self, ind_to_predicates):
        """
        获取目标谓词在完整谓词列表中的索引
        """
        target_indices = []
        for pred_name in self.target_predicates:
            if pred_name in ind_to_predicates:
                idx = ind_to_predicates.index(pred_name)
                target_indices.append(idx)

        return target_indices

    def convert_to_global_logits(self, dpl_rel_dists, pred_indices, global_rel_logits):
        """
        将DPL的预测结果转换到全局的51维谓词空间

        Args:
            dpl_rel_dists: DPL预测的分数 [num_relations, num_target_predicates]
            pred_indices: 目标谓词在全局空间中的索引
            global_rel_logits: 全局的原始预测 [num_relations, 51]

        Returns:
            updated_global_logits: 更新后的全局预测
        """
        if pred_indices is None or len(pred_indices) == 0:
            return global_rel_logits

        updated_logits = global_rel_logits.clone()

        # 将DPL的预测结果更新到对应的谓词位置
        for i, global_idx in enumerate(pred_indices):
            if global_idx < global_rel_logits.shape[1]:
                # 使用DPL预测替换对应谓词的分数
                updated_logits[:, global_idx] = dpl_rel_dists[:, i]

        return updated_logits


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


def test_predicate_refiner():
    """测试 DPL 谓词精调器"""
    print("Testing DPL PredicateRefiner...")

    # 测试参数
    batch_size = 5
    feature_dim = 4096
    prototype_dim = 128

    # 创建测试数据
    features = torch.randn(batch_size, feature_dim).to(CUDA_DEVICE)
    target_labels = torch.randint(0, 10, (batch_size,)).to(CUDA_DEVICE)  # 目标谓词索引 (0-9)

    # 创建 DPL 精调器
    refiner = PredicateRefiner(
        feature_dim=feature_dim,
        prototype_dim=prototype_dim,
        avg_sample_size=10,
        alpha=5.0,
        radius=0.5
    ).to(CUDA_DEVICE)

    # 测试推理模式
    print("\n=== Inference Mode Test ===")
    refiner.eval()
    with torch.no_grad():
        rel_dists, confidences = refiner(features)

    print(f"Input features shape: {features.shape}")
    print(f"Output rel_dists shape: {rel_dists.shape}")
    print(f"Confidences shape: {confidences.shape}")
    print(f"Target predicates: {refiner.target_predicates}")
    print(f"Sample sizes: {refiner.sample_size_array}")
    print(f"Confidence range: [{confidences.min().item():.3f}, {confidences.max().item():.3f}]")

    # 测试训练模式
    print("\n=== Training Mode Test ===")
    refiner.train()
    rel_dists_train, loss_dict = refiner(features, target_labels=target_labels)

    print(f"Training rel_dists shape: {rel_dists_train.shape}")
    print(f"Losses: {[(k, v.item()) for k, v in loss_dict.items()]}")

    # 验证原型嵌入
    print(f"\nPrototype embedding shape: {refiner.prototype_emb.shape}")
    print(f"Gaussian embedding output shape: {refiner.gaussian_emb(refiner.prototype_emb).shape}")

    # 测试全局logits转换
    global_logits = torch.randn(batch_size, 51).to(CUDA_DEVICE)
    target_indices = refiner.get_target_predicate_indices(list(refiner.predicate_dict_loader.idx_to_predicate.values()))
    updated_logits = refiner.convert_to_global_logits(rel_dists, target_indices, global_logits)

    print(f"Global logits shape: {global_logits.shape}")
    print(f"Updated logits shape: {updated_logits.shape}")
    print(f"Target predicate indices: {target_indices}")

    print("DPL PredicateRefiner test completed successfully!")


if __name__ == "__main__":
    test_predicate_refiner()