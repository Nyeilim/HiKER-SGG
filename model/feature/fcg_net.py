import torch
import torch.nn.functional as fn
from torch.cuda import current_device
from torch.nn import Module, Linear, MultiheadAttention
from time import time as time_time

from model.feature.fcg_helper import hierarchical_reasoning_fcg, pre_process, post_process, pred_center_reasoning, \
    filter_out_nonrel
from model.feature.fcg_builder import FCGBuilder
from model.util import MLP
from config import logger

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')


class FCGNet(Module):
    """
    基于细粒度知识图(FCG)的场景图生成网络
    """

    def __init__(self, time_step_num=3, hidden_dim=1024):
        super(FCGNet, self).__init__()
        self.time_step_num = time_step_num
        self.hidden_dim = hidden_dim

        # 加载 FCG
        self.fcg = FCGBuilder(hidden_dim=hidden_dim)

        # 初始化消息传递所需的 MLP 层 @formatter:off
        self.mlp_send_l1_nodes = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_send_l2_nodes = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_send_l3_nodes = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_send_triplet = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_rcv_l1_nodes = MLP([2 * hidden_dim // 4, 2 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.mlp_rcv_l2_nodes = MLP([2 * hidden_dim // 4, 2 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.mlp_rcv_l3_nodes = MLP([hidden_dim // 4, hidden_dim // 2, hidden_dim], act_fn='ReLU', last_act=True)
        self.mlp_rcv_triplet = MLP([hidden_dim // 4, hidden_dim // 2, hidden_dim], act_fn='ReLU', last_act=True)
        # @formatter:on

        # 初始化 GRU 规则所需的线性层，eq3/4/5 代表 GRU 论文中的三条核心等式，w/u 作用于输入/隐状态的权重
        ## l1
        self.fc_eq3_w_l1 = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_l1 = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_l1 = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_l1 = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_l1 = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_l1 = Linear(hidden_dim, hidden_dim)
        ## l2_nodes
        self.fc_eq3_w_l2 = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_l2 = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_l2 = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_l2 = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_l2 = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_l2 = Linear(hidden_dim, hidden_dim)
        ## l3_nodes
        self.fc_eq3_w_l3 = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_l3 = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_l3 = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_l3 = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_l3 = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_l3 = Linear(hidden_dim, hidden_dim)
        ## triplet
        self.fc_eq3_w_tri = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_tri = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_tri = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_tri = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_tri = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_tri = Linear(hidden_dim, hidden_dim)

        # 初始化输出投影层
        self.fc_triplet_logits = Linear(hidden_dim, hidden_dim)
        self.fc_fcg_logits = Linear(hidden_dim, hidden_dim)

    def forward(self, rel_inds, ent_probs, vr):
        """
        FCG Net 的前向传播
        :param rel_inds: shape(img_all_rels,2) <s,o> 二元组
        :param ent_probs: shape(img_gt_boxes,151) boxes 的类别概率分布
        :param vr: shape(img_all_rels,1024) 关系的视觉特征
        :return: pred_cls_score: 谓词的预测概率
                scpred_cls_score: 超类谓词的预测概率
        """
        # 切断与原计算图的联系，只有传过来的 vr 是连接着计算图的，切断
        triplet = vr.clone().detach()  # 使用关系视觉特征作为三元组特征

        # 复制 FCG 节点特征
        fcg_l1_feats = torch.stack([node.feat for node in self.fcg.l1_nodes]).to(CUDA_DEVICE)
        fcg_l2_feats = torch.stack([node.feat for node in self.fcg.l2_nodes]).to(CUDA_DEVICE)
        fcg_l3_feats = torch.stack([node.feat for node in self.fcg.l3_nodes]).to(CUDA_DEVICE)

        # 复制 FCG 边权重
        fcg_edges_l2_l3 = self.fcg.edges_l2_l3.to(CUDA_DEVICE)
        fcg_edges_l1_l2 = self.fcg.edges_l1_l2.to(CUDA_DEVICE)
        fcg_edges_l3_l2 = fcg_edges_l2_l3.t()
        fcg_edges_l2_l1 = fcg_edges_l1_l2.t()

        # 预处理
        bridge_edges_tri_l1, normal_rel_mask, triplet = pre_process(self.fcg, rel_inds, ent_probs, triplet)
        bridge_edges_l1_tri = bridge_edges_tri_l1.t()

        for t in range(self.time_step_num):
            # 发出消息
            msg_send_l1_nodes = self.mlp_send_l1_nodes(fcg_l1_feats)
            msg_send_l2_nodes = self.mlp_send_l2_nodes(fcg_l2_feats)
            msg_send_l3_nodes = self.mlp_send_l3_nodes(fcg_l3_feats)
            msg_send_triplet = self.mlp_send_triplet(triplet)

            # 接收消息
            msg_rcv_l1_nodes = self.mlp_rcv_l1_nodes(torch.cat([
                torch.mm(fcg_edges_l1_l2, msg_send_l2_nodes),
                torch.mm(bridge_edges_l1_tri, msg_send_triplet),
            ], dim=1))
            msg_rcv_l2_nodes = self.mlp_rcv_l2_nodes(torch.cat([
                torch.mm(fcg_edges_l2_l3, msg_send_l3_nodes),
                torch.mm(fcg_edges_l2_l1, msg_send_l1_nodes),
            ], dim=1))
            msg_rcv_l3_nodes = self.mlp_rcv_l3_nodes(torch.cat([
                torch.mm(fcg_edges_l3_l2, msg_send_l2_nodes),
            ], dim=1))
            msg_rcv_triplet = self.mlp_rcv_triplet(torch.cat([
                torch.mm(bridge_edges_tri_l1, msg_send_l1_nodes),
            ], dim=1))

            # 释放引用
            del msg_send_l1_nodes, msg_send_l2_nodes, msg_send_l3_nodes, msg_send_triplet

            # 更新节点特征
            z_l1 = torch.sigmoid(self.fc_eq3_w_l1(msg_rcv_l1_nodes) + self.fc_eq3_u_l1(fcg_l1_feats))
            r_l1 = torch.sigmoid(self.fc_eq4_w_l1(msg_rcv_l1_nodes) + self.fc_eq4_u_l1(fcg_l1_feats))
            h_l1 = torch.tanh(self.fc_eq5_w_l1(msg_rcv_l1_nodes) + self.fc_eq5_u_l1(r_l1 * fcg_l1_feats))
            fcg_l1_feats = (1 - z_l1) * fcg_l1_feats + z_l1 * h_l1
            del msg_rcv_l1_nodes, r_l1, z_l1, h_l1

            z_l2 = torch.sigmoid(self.fc_eq3_w_l2(msg_rcv_l2_nodes) + self.fc_eq3_u_l2(fcg_l2_feats))
            r_l2 = torch.sigmoid(self.fc_eq4_w_l2(msg_rcv_l2_nodes) + self.fc_eq4_u_l2(fcg_l2_feats))
            h_l2 = torch.tanh(self.fc_eq5_w_l2(msg_rcv_l2_nodes) + self.fc_eq5_u_l2(r_l2 * fcg_l2_feats))
            fcg_l2_feats = (1 - z_l2) * fcg_l2_feats + z_l2 * h_l2
            del msg_rcv_l2_nodes, r_l2, z_l2, h_l2

            z_l3 = torch.sigmoid(self.fc_eq3_w_l3(msg_rcv_l3_nodes) + self.fc_eq3_u_l3(fcg_l3_feats))
            r_l3 = torch.sigmoid(self.fc_eq4_w_l3(msg_rcv_l3_nodes) + self.fc_eq4_u_l3(fcg_l3_feats))
            h_l3 = torch.tanh(self.fc_eq5_w_l3(msg_rcv_l3_nodes) + self.fc_eq5_u_l3(r_l3 * fcg_l3_feats))
            fcg_l3_feats = (1 - z_l3) * fcg_l3_feats + z_l3 * h_l3
            del msg_rcv_l3_nodes, r_l3, z_l3, h_l3

            z_tri = torch.sigmoid(self.fc_eq3_w_tri(msg_rcv_triplet) + self.fc_eq3_u_tri(triplet))
            r_tri = torch.sigmoid(self.fc_eq4_w_tri(msg_rcv_triplet) + self.fc_eq4_u_tri(triplet))
            h_tri = torch.tanh(self.fc_eq5_w_tri(msg_rcv_triplet) + self.fc_eq5_u_tri(r_tri * triplet))
            triplet = (1 - z_tri) * triplet + z_tri * h_tri
            del msg_rcv_triplet, r_tri, z_tri, h_tri

            # 使用 FC 层而不是强制归一化特征
            triplet = self.fc_triplet_logits(triplet)
            fcg_l1_feats = self.fc_fcg_logits(fcg_l1_feats)
            fcg_l2_feats = self.fc_fcg_logits(fcg_l2_feats)
            fcg_l3_feats = self.fc_fcg_logits(fcg_l3_feats)

        pred_cls_score = hierarchical_reasoning_fcg(self.fcg, bridge_edges_tri_l1, triplet, fcg_l2_feats, fcg_l3_feats)
        pred_cls_score = post_process(pred_cls_score, normal_rel_mask)

        return pred_cls_score


class FCGNetV2(Module):
    """
    基于细粒度知识图(FCG)的场景图生成网络 V2 版本
    使用交叉注意力机制对齐特征，而不是通过消息传递更新FCG节点
    """

    def __init__(self, hidden_dim=1024, num_heads=8):
        super(FCGNetV2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 加载 FCG
        self.fcg = FCGBuilder(hidden_dim=hidden_dim)

        # 构建并缓存所有需要的张量，使用register_buffer让PyTorch自动处理设备迁移
        fcg_l2_feats = torch.stack([node.feat for node in self.fcg.l2_nodes])
        fcg_l3_feats = torch.stack([node.feat for node in self.fcg.l3_nodes])
        
        # 构建谓词中心特征张量，按照 key 从小到大排序后转换为 list
        sorted_pred_centers = [self.fcg.pred_center[k] for k in sorted(self.fcg.pred_center.keys())]
        pred_centers = torch.stack(sorted_pred_centers)
        
        # 注册所有buffer，让PyTorch自动处理设备迁移和保存/加载
        self.register_buffer('fcg_l2_feats', fcg_l2_feats)
        self.register_buffer('fcg_l3_feats', fcg_l3_feats)
        self.register_buffer('pred_centers', pred_centers)

        # 使用PyTorch现成的MultiheadAttention替换手动实现的交叉注意力
        self.cross_attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 层归一化
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.ffn = MLP([hidden_dim, hidden_dim * 2, hidden_dim], act_fn='ReLU', last_act=False)

    def forward(self, rel_inds, ent_probs, vr):
        """
        FCGNetV2 的前向传播
        :param rel_inds: shape(img_all_rels,2) <s,o> 二元组
        :param ent_probs: shape(img_gt_boxes,151) boxes 的类别概率分布
        :param vr: shape(img_all_rels,1024) 关系的视觉特征
        :return: pred_cls_score: 谓词的预测概率
        """
        # 预处理
        _start = time_time()
        normal_rel_mask, triplet = filter_out_nonrel(self.fcg, rel_inds, ent_probs, vr)
        _pre_process = time_time()

        # 使用PyTorch的MultiheadAttention进行交叉注意力
        # 输入形状: (batch_size, seq_len, hidden_dim)
        # triplet: (num_rels, hidden_dim) -> (1, num_rels, hidden_dim)
        # pred_centers: (num_preds, hidden_dim) -> (1, num_preds, hidden_dim)
        triplet_expanded = triplet.unsqueeze(0)  # (1, num_rels, hidden_dim)
        pred_centers_expanded = self.pred_centers.unsqueeze(0)  # (1, num_preds, hidden_dim)
        
        # 交叉注意力：triplet作为query，pred_centers作为key和value
        attn_output, _ = self.cross_attention(
            query=triplet_expanded,
            key=pred_centers_expanded,
            value=pred_centers_expanded
        )
        
        # 残差连接和层归一化
        aligned_triplet = self.norm1(triplet_expanded + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(aligned_triplet)
        
        # 残差连接和层归一化
        aligned_triplet = self.norm2(aligned_triplet + ffn_output)
        
        # 恢复原始形状
        aligned_triplet = aligned_triplet.squeeze(0)  # (num_rels, hidden_dim)

        # 使用对齐后的特征进行层级推理
        pred_cls_score = pred_center_reasoning(self.pred_centers, aligned_triplet)
        pred_cls_score = post_process(pred_cls_score, normal_rel_mask)

        _end = time_time()
        logger.debug(f"FCGNet 前向传播总耗时:{_end - _start:.4f}s, 预处理耗时:{_pre_process - _start:.4f}s")

        return pred_cls_score
