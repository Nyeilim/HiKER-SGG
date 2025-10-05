"""
DPL (Diversity-aware Prototype-based Learning) Classifier
独立的原型分类器模块，用于缓解场景图生成中的长尾问题

参考论文: https://arxiv.org/abs/2407.15396
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear

from config import MODEL


class DPLClassifier(nn.Module):
    """
    基于原型学习的独立分类器
    与 FCGNetV2 类似的独立模块设计
    """

    def __init__(self, input_dim=1024, num_rel_cls=51, hidden_dim=128):
        super(DPLClassifier, self).__init__()

        self.input_dim = input_dim
        self.num_rel_cls = num_rel_cls
        self.dpl_dim = hidden_dim

        print(f'[DPLClassifier] Initializing with input_dim={input_dim}, num_rel_cls={num_rel_cls}, dpl_dim={hidden_dim}')

        # DPL 特征压缩层 (1024 -> 128)
        self.rel_compress = Linear(input_dim, self.dpl_dim)

        # 原型嵌入 (每个谓词类一个原型)
        self.proto_emb = nn.Parameter(
            torch.zeros(num_rel_cls, self.dpl_dim),
            requires_grad=True
        )
        nn.init.orthogonal_(self.proto_emb)

        # 高斯参数网络 (学习每个原型的方差)
        self.gaussian_emb = Linear(self.dpl_dim, self.dpl_dim * 2)

        # 距离转换参数
        self.shift = nn.Parameter(torch.ones(1) * 15.0)
        self.negative_scale = nn.Parameter(torch.ones(1) * 15.0)

        # DPL 超参数
        self.dpl_sample_size = MODEL.DPL.AVG_NUM_SAMPLE
        self.dpl_alpha = MODEL.DPL.ALPHA
        self.dpl_radius = MODEL.DPL.RADIUS

        # 采样数量数组（可选：根据类别频率调整）
        if MODEL.DPL.FREQ_BASED_DIFF_N:
            print('[DPLClassifier] FREQ_BASED_DIFF_N enabled but not implemented yet, using uniform sampling')
            self.sample_size_array = np.ones(num_rel_cls, dtype=int) * self.dpl_sample_size
        else:
            self.sample_size_array = np.ones(num_rel_cls, dtype=int) * self.dpl_sample_size

    def forward(self, vr, rel_labels=None):
        """
        DPL 分类器前向传播

        Args:
            vr: (num_rels, 1024) 关系视觉特征
            rel_labels: (num_rels,) 关系标签，仅训练时提供

        Returns:
            dpl_logits: (num_rels, num_rel_cls) DPL 分类 logits
            dpl_losses: dict, 包含 'ortho_loss' 和 'sample_loss'
        """
        # 压缩特征到 DPL 维度
        dpl_features = self.rel_compress(vr)  # (num_rels, dpl_dim)

        # 【关键】归一化特征到单位球面
        dpl_features = dpl_features / (dpl_features.norm(dim=1, keepdim=True) + 1e-8)

        # 计算 DPL logits 和损失
        dpl_logits, dpl_losses = self._compute_dpl(dpl_features, rel_labels)

        return dpl_logits, dpl_losses

    def _compute_dpl(self, features, rel_labels=None):
        """
        计算 DPL 的 logits 和损失

        Args:
            features: (num_rel, dpl_dim) 归一化后的关系特征
            rel_labels: (num_rel,) 关系标签，仅训练时提供

        Returns:
            dpl_logits: (num_rel, num_rel_cls) DPL 分类 logits
            add_losses: dict, 包含 'ortho_loss' 和 'sample_loss'
        """
        add_losses = {}

        # 归一化原型
        predicate_proto_norm = self.proto_emb / (self.proto_emb.norm(dim=1, keepdim=True) + 1e-8)

        # 计算高斯参数
        gaussian = self.gaussian_emb(predicate_proto_norm)
        mu, logsigma = torch.split(gaussian, self.dpl_dim, dim=1)

        # 计算特征到原型的距离
        num_rel = features.size(0)
        rel_rep_expand = features.unsqueeze(1).expand(-1, self.proto_emb.size(0), -1)
        proto_expand = predicate_proto_norm.unsqueeze(0).expand(num_rel, -1, -1)
        distance_set = (rel_rep_expand - proto_expand).norm(dim=2)

        # 训练时：简单距离转换为 logits
        dpl_logits = -self.negative_scale * distance_set + self.shift

        # ========== 训练阶段：计算额外损失 ==========
        if self.training and rel_labels is not None:
            # 1. 正交损失：确保不同类别原型相互正交
            proto_sim = torch.matmul(predicate_proto_norm, predicate_proto_norm.t())
            ortho_loss = self._get_orth_loss(proto_sim)
            add_losses['ortho_loss'] = ortho_loss

            # 2. 采样损失：从高斯分布采样，强制特征落在原型半径内
            detach_proto = predicate_proto_norm.detach()
            z = self._sample_gaussian_tensors(
                detach_proto, logsigma, self.sample_size_array
            ).view(-1, self.dpl_dim)

            # 计算到采样点的距离
            distance_set_z = self._distance(features, z)

            # 找到每个类别的最小距离
            distance_set_m = self._get_min_dists_z(distance_set_z, self.sample_size_array)

            # 选择正确类别的距离
            selected_distance = distance_set_m[torch.arange(rel_labels.size(0)), rel_labels]

            # Hinge loss: 距离应该在半径内
            zeros_tensor = torch.zeros_like(selected_distance)
            sample_loss = torch.mean(
                torch.where(
                    selected_distance > self.dpl_radius,
                    torch.pow(selected_distance - self.dpl_radius, 2),
                    zeros_tensor
                )
            )
            add_losses['sample_loss'] = sample_loss * self.dpl_alpha

        # ========== 测试阶段：使用方差加权距离 ==========
        elif not self.training:
            # 限制方差范围，避免数值不稳定
            sigma = torch.clamp(logsigma.exp(), min=0.1, max=10.0)

            # 方差加权距离
            nd = (rel_rep_expand - proto_expand) / (sigma.unsqueeze(0) + 1e-8)
            nd = nd.norm(dim=2)

            # 归一化
            ndn = (nd.t() / (nd.max(dim=1)[0] + 1e-8) * distance_set.max(dim=1)[0]).t()
            dpl_logits = -self.negative_scale * ndn + self.shift

        return dpl_logits, add_losses

    def _get_orth_loss(self, proto_sim):
        """计算原型正交损失"""
        eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
        loss_orth = torch.abs(proto_sim[eye_sim == 1]).mean()
        return loss_orth

    def _sample_gaussian_tensors(self, mu, logsigma, num_samples):
        """从高斯分布采样"""
        total_samples = self._sample_each(mu[0], logsigma[0], num_samples[0])
        for i in range(1, mu.size(0)):
            samples = self._sample_each(mu[i], logsigma[i], num_samples[i])
            total_samples = torch.cat([total_samples, samples], dim=0)
        return total_samples

    def _sample_each(self, mu_i, logsigma_i, num_samples_i):
        """为单个类别采样"""
        eps = torch.randn(num_samples_i, mu_i.size(0), dtype=mu_i.dtype, device=mu_i.device)
        samples = eps.mul(torch.exp(logsigma_i)).add_(mu_i)
        return samples

    def _distance(self, t1, t2):
        """计算欧氏距离矩阵"""
        t1_square = torch.sum(t1 ** 2, dim=1, keepdim=True)
        t2_square = torch.sum(t2 ** 2, dim=1)
        distance_set = torch.sqrt(
            t1_square + t2_square - 2 * torch.matmul(t1, t2.t()) + 1e-8
        )
        return distance_set

    def _get_min_dists_z(self, rel_dists_z, num_samples):
        """获取到采样点的最小距离"""
        total_md = rel_dists_z[:, :num_samples[0]].sort(1, descending=False)[0][:, :1]
        count = num_samples[0]
        for i in range(1, len(num_samples)):
            md = rel_dists_z[:, count:count+num_samples[i]].sort(1, descending=False)[0][:, :1]
            total_md = torch.cat([total_md, md], dim=1)
            count = count + num_samples[i]
        return total_md
