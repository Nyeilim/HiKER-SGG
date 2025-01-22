import torch
from torch import nn
import torch.nn.functional as F
import math

class FCGNet(nn.Module):
    """细粒度三元组分类网络"""
    
    def __init__(self, fcg_builder, hidden_dim=1024):
        """
        Args:
            fcg_builder: FCG构建器实例
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.fcg = fcg_builder
        self.hidden_dim = hidden_dim
        
        # 投影层,将输入特征投影到统一空间
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        # GRU用于信息传递
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim, len(self.fcg.level3_nodes))
        
        # 虚节点惩罚系数
        self.virtual_penalty = 0.5
        
    def forward(self, sp_feats, se_probs):
        """
        Args:
            sp_feats: SP节点特征 [batch_size, hidden_dim]
            se_probs: SE对应的概率分布 [batch_size, 2, 151]
        Returns:
            triplet_probs: 三元组概率分布
        """
        batch_size = sp_feats.size(0)
        
        # 1. 特征投影
        x = self.proj(sp_feats)
        
        # 2. 两次信息传递
        for _ in range(2):
            # Level 1 -> Level 2
            msg_l1_l2 = torch.matmul(x, self.fcg.edges_l1_l2)
            
            # Level 2 -> Level 3  
            msg_l2_l3 = torch.matmul(msg_l1_l2, self.fcg.edges_l2_l3)
            
            # GRU更新
            x = self.gru(msg_l2_l3.unsqueeze(0))[0].squeeze(0)
            
        # 3. 分类预测
        logits = self.classifier(x)
        
        # 4. 对虚节点进行惩罚
        virtual_mask = torch.zeros_like(logits)
        for i, (s,p,o) in enumerate(self.fcg.level3_nodes):
            if self.fcg.level3_virtual[i]:  # 虚节点
                virtual_mask[:,i] = self.virtual_penalty
                
        logits = logits - virtual_mask
        
        # 5. 层级分类概率计算
        # 计算两个分支的概率
        subj_branch_probs = self.compute_subject_branch_probs(logits, se_probs)  # <person,x,x> 分支
        obj_branch_probs = self.compute_object_branch_probs(logits, se_probs)    # <x,x,snow> 分支
        
        # 计算分支权重并融合
        branch_weights = self.compute_branch_weights(subj_branch_probs, obj_branch_probs)
        final_probs = (
            branch_weights[:, 0:1] * subj_branch_probs + 
            branch_weights[:, 1:2] * obj_branch_probs
        )
        
        return final_probs

    def compute_subject_branch_probs(self, logits, se_probs):
        """计算主语分支概率"""
        # 第一级:主语实体概率
        level1_probs = se_probs[:, 0]  # [batch_size, 151]
        
        # 第二级:<主语,谓词>模式概率
        level2_sim = F.cosine_similarity(
            logits.unsqueeze(1),
            self.fcg.level2_feats[self.fcg.subj_patterns].unsqueeze(0),
            dim=2
        )
        level2_probs = F.softmax(level2_sim, dim=1)
        
        # 第三级:具体三元组概率
        level3_probs = F.softmax(logits, dim=1)
        
        # 合并三级概率
        return level1_probs * level2_probs * level3_probs

    def compute_object_branch_probs(self, logits, se_probs):
        """计算宾语分支概率"""
        # 第一级:宾语实体概率
        level1_probs = se_probs[:, 1]  # [batch_size, 151]
        
        # 第二级:<谓词,宾语>模式概率
        level2_sim = F.cosine_similarity(
            logits.unsqueeze(1),
            self.fcg.level2_feats[self.fcg.obj_patterns].unsqueeze(0),
            dim=2
        )
        level2_probs = F.softmax(level2_sim, dim=1)
        
        # 第三级:具体三元组概率
        level3_probs = F.softmax(logits, dim=1)
        
        # 合并三级概率
        return level1_probs * level2_probs * level3_probs

    def compute_branch_weights(self, subj_probs, obj_probs):
        """计算两个分支的权重"""
        # 基于分支的置信度计算权重
        subj_confidence = self.compute_branch_confidence(subj_probs)
        obj_confidence = self.compute_branch_confidence(obj_probs)
        
        # Softmax确保权重和为1
        weights = torch.stack([subj_confidence, obj_confidence], dim=1)
        weights = F.softmax(weights, dim=1)
        
        return weights

    def compute_branch_confidence(self, branch_probs):
        """计算分支的置信度"""
        # 使用熵来衡量置信度
        entropy = -(branch_probs * torch.log(branch_probs + 1e-10)).sum(dim=1)
        confidence = 1 - entropy/math.log(branch_probs.size(1))  # 归一化的置信度
        return confidence

    def loss(self, pred_probs, target_indices, rel_class_weights=None):
        """
        计算分类损失,参考hiker_model.py中的rel_loss
        Args:
            pred_probs: 预测概率 [batch_size, num_triplets] 
            target_indices: 目标三元组索引 [batch_size]
            rel_class_weights: 关系类别权重
        """
        if rel_class_weights is not None:
            return F.nll_loss(torch.log(pred_probs + 1e-10), 
                            target_indices,
                            weight=rel_class_weights)
        else:
            return F.nll_loss(torch.log(pred_probs + 1e-10), 
                            target_indices) 