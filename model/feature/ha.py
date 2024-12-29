import torch
import torch.nn as nn
import torch.nn.functional as F

class HA(nn.Module):
    """
    分层注意力模块
    """
    def __init__(self, hidden_dim=512, num_heads=8):
        """
        初始化 HA 模块
        :param hidden_dim: 隐藏层维度
        :param num_heads: 注意力头数
        """
        super(HA, self).__init__()
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, X, Y):
        """
        前向传播
        :param X: 输入特征 X [num_entities, batch_size, hidden_dim]
        :param Y: 输入特征 Y [num_entities, batch_size, hidden_dim]
        :return: 更新后的特征 X, Y
        """
        # 自注意力
        X_sa, _ = self.self_attn(X, X, X)
        Y_sa, _ = self.self_attn(Y, Y, Y)
        
        # 交叉注意力
        X_ca, _ = self.cross_attn(X, Y, Y)
        Y_ca, _ = self.cross_attn(Y, X, X)
        
        # 使用 1/2 系数融合特征
        X_updated = 0.5 * (X_sa + X_ca)
        Y_updated = 0.5 * (Y_sa + Y_ca)
        
        return X_updated, Y_updated


class DoubleHA(nn.Module):
    """
    双层分层注意力模块
    """
    def __init__(self, hidden_dim=512, num_heads=8):
        """
        初始化双层 HA 模块
        :param hidden_dim: 隐藏层维度
        :param num_heads: 注意力头数
        """
        super(DoubleHA, self).__init__()
        
        # 两层 HA
        self.ha1 = HA(hidden_dim, num_heads)
        self.ha2 = HA(hidden_dim, num_heads)

    def forward(self, X, Y):
        """
        前向传播
        :param X: 输入特征 X [num_entities, batch_size, hidden_dim]
        :param Y: 输入特征 Y [num_entities, batch_size, hidden_dim]
        :return: 更新后的特征 X, Y
        """
        # 第一层 HA
        X, Y = self.ha1(X, Y)
        
        # 第二层 HA
        X, Y = self.ha2(X, Y)
        
        return X, Y