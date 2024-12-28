import torch
import torch.nn as nn
import torch.nn.functional as F

class DRM(nn.Module):
    """
    双粒度关系建模模块
    """
    def __init__(self, hidden_dim=512, num_heads=8):
        """
        初始化 DRM 模块
        :param hidden_dim: 隐藏层维度
        :param num_heads: 注意力头数
        """
        super(DRM, self).__init__()
        
        # 实体级别的注意力
        self.entity_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # 谓词级别的注意力
        self.predicate_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # 实体到谓词的映射
        self.entity_to_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 谓词到实体的映射
        self.pred_to_entity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, entity_feats, pred_feats, rel_inds):
        """
        前向传播
        :param entity_feats: 实体特征 [num_entities, hidden_dim]
        :param pred_feats: 谓词特征 [num_predicates, hidden_dim]
        :param rel_inds: 关系索引 [num_rels, 2]
        :return: 增强后的谓词特征
        """
        # 实体级别的注意力
        entity_feats = entity_feats.unsqueeze(0)  # [1, num_entities, hidden_dim] 添加维度 0
        entity_attn_out, _ = self.entity_attn(entity_feats, entity_feats, entity_feats)
        entity_attn_out = entity_attn_out.squeeze(0)  # [num_entities, hidden_dim]
        
        # 谓词级别的注意力
        pred_feats = pred_feats.unsqueeze(0)  # [1, num_predicates, hidden_dim]
        pred_attn_out, _ = self.predicate_attn(pred_feats, pred_feats, pred_feats)
        pred_attn_out = pred_attn_out.squeeze(0)  # [num_predicates, hidden_dim]
        
        # 获取主语和宾语的特征
        subj_feats = entity_attn_out[rel_inds[:, 0]]  # [num_rels, hidden_dim]
        obj_feats = entity_attn_out[rel_inds[:, 1]]   # [num_rels, hidden_dim]
        
        # 实体到谓词的映射
        entity_context = self.entity_to_pred(subj_feats + obj_feats)  # [num_rels, hidden_dim]
        
        # 谓词到实体的映射
        pred_context = self.pred_to_entity(pred_attn_out)  # [num_predicates, hidden_dim]
        
        # 特征融合
        enhanced_feats = self.fusion(torch.cat([entity_context, pred_context], dim=-1))
        
        return enhanced_feats 