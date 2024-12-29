import torch
import torch.nn.functional as F
import numpy as np

class DecomposedPrior:
    """分解的统计先验，将P(p|s,o)分解为P(p|s)和P(p|o)"""
    def __init__(self, edge_matrix):
        """
        Args:
            edge_matrix: [151, 151, 51] 原始的统计先验矩阵
        """
        # 转换为PyTorch张量并移至GPU
        self.edge_matrix = torch.from_numpy(edge_matrix).float().cuda()
        
        # 计算P(p|s)：对o维度求和后归一化
        self.subj_pred_dist = self.edge_matrix.sum(1)  # [151, 51]
        self.subj_pred_dist = F.normalize(self.subj_pred_dist, p=1, dim=1)
        
        # 计算P(p|o)：对s维度求和后归一化
        self.obj_pred_dist = self.edge_matrix.sum(0)   # [151, 51]
        self.obj_pred_dist = F.normalize(self.obj_pred_dist, p=1, dim=1)
        
        # 计算P(p)：边缘分布
        self.pred_marginal = self.edge_matrix.sum((0,1))  # [51]
        self.pred_marginal = F.normalize(self.pred_marginal, p=1, dim=0)
        
    def get_decomposed_prior(self, rel_inds, obj_labels):
        """
        计算分解的先验概率
        
        Args:
            rel_inds: [num_rels, 2] 关系中的实体对索引
            obj_labels: [num_objs] 实体的类别标签
            
        Returns:
            prior: [num_rels, 51] 分解后的先验概率
        """
        # 获取实体对的类别标签
        s_labels = obj_labels[rel_inds[:, 0]]  # [num_rels]
        o_labels = obj_labels[rel_inds[:, 1]]  # [num_rels]
        
        # 获取条件概率
        p_pred_s = self.subj_pred_dist[s_labels]  # [num_rels, 51]
        p_pred_o = self.obj_pred_dist[o_labels]   # [num_rels, 51]
        
        # 使用条件独立性假设计算联合概率
        # P(p|s,o) ≈ P(p|s) * P(p|o) / P(p)
        prior = (p_pred_s * p_pred_o) / (self.pred_marginal + 1e-8)
        
        # 归一化
        prior = F.normalize(prior, p=1, dim=1)
        return prior

class ContextAwarePrior:
    """考虑图片级上下文的先验"""
    def __init__(self, edge_matrix):
        self.edge_matrix = torch.from_numpy(edge_matrix).float().cuda()
        # 计算分解的条件概率
        self.subj_pred_dist = self.edge_matrix.sum(1)  # [151, 51]
        self.obj_pred_dist = self.edge_matrix.sum(0)   # [151, 51]
        self.pred_marginal = self.edge_matrix.sum((0,1))  # [51]
        
    def get_context_aware_prior(self, rel_inds, obj_labels):
        """
        计算上下文感知的先验概率
        
        Args:
            rel_inds: [num_rels, 2] 关系中的实体对索引
            obj_labels: [num_objs] 实体的类别标签
            
        Returns:
            prior: [num_rels, 51] 考虑上下文的先验概率
        """
        # 计算图片中实体对的共现次数
        cooccur = torch.zeros(151, 151, device=obj_labels.device)
        for s_idx, o_idx in rel_inds:
            s_label = obj_labels[s_idx]
            o_label = obj_labels[o_idx]
            cooccur[s_label, o_label] += 1
        
        priors = []
        # 对每个关系
        for rel in rel_inds:
            s, o = obj_labels[rel[0]], obj_labels[rel[1]]
            
            if self.edge_matrix[s, o].sum() == 0:
                # 如果实体对在训练集中从未出现过，说明它们之间不应该有关系
                prior = torch.zeros(51, device=obj_labels.device)
                prior[0] = 1.0  # 将所有概率分配给背景类（假设索引0是背景类）
            elif cooccur[s, o] > 1:
                # 如果实体对在图片中频繁共现，使用它们的联合概率
                prior = self.edge_matrix[s, o]
            else:
                # 否则使用分解的条件概率来缓解数据稀疏问题
                prior = (self.subj_pred_dist[s] * self.obj_pred_dist[o]) / (self.pred_marginal + 1e-8)
            
            priors.append(F.normalize(prior, p=1, dim=0))
        
        return torch.stack(priors) 