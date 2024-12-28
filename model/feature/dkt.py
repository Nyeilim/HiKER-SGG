import torch
import torch.nn as nn
import torch.nn.functional as F

class DKT(nn.Module):
    """
    双粒度知识迁移模块
    """
    def __init__(self, hidden_dim=512, num_rel_classes=51):
        """
        初始化 DKT 模块
        :param hidden_dim: 隐藏层维度
        :param num_rel_classes: 关系类别数
        """
        super(DKT, self).__init__()
        
        # 知识迁移网络
        self.knowledge_transfer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 关系分类器
        self.rel_classifier = nn.Linear(hidden_dim, num_rel_classes)
        
        # 知识蒸馏温度
        self.temperature = 2.0
        
        # 知识融合层
        self.knowledge_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, student_feats, teacher_feats):
        """
        前向传播
        :param student_feats: 学生模型特征 [batch_size, hidden_dim]
        :param teacher_feats: 教师模型特征 [batch_size, hidden_dim]
        :return: 知识迁移后的特征和预测结果
        """
        # 知识迁移
        transferred_knowledge = self.knowledge_transfer(teacher_feats)
        
        # 注意力机制增强
        student_feats = student_feats.unsqueeze(0)  # [1, batch_size, hidden_dim]
        transferred_knowledge = transferred_knowledge.unsqueeze(0)  # [1, batch_size, hidden_dim]
        attn_out, _ = self.attention(student_feats, transferred_knowledge, transferred_knowledge)
        attn_out = attn_out.squeeze(0)  # [batch_size, hidden_dim]
        
        # 知识融合
        fused_features = self.knowledge_fusion(
            torch.cat([attn_out, student_feats.squeeze(0)], dim=-1)
        )
        
        # 关系分类
        logits = self.rel_classifier(fused_features)
        
        # 知识蒸馏
        student_probs = F.softmax(logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(self.rel_classifier(teacher_feats) / self.temperature, dim=-1)
        
        return {
            'logits': logits,
            'student_probs': student_probs,
            'teacher_probs': teacher_probs,
            'fused_features': fused_features
        }
    
    def compute_kd_loss(self, student_probs, teacher_probs):
        """
        计算知识蒸馏损失
        :param student_probs: 学生模型概率分布
        :param teacher_probs: 教师模型概率分布
        :return: 知识蒸馏损失
        """
        return F.kl_div(
            F.log_softmax(student_probs / self.temperature, dim=-1),
            F.softmax(teacher_probs / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2) 