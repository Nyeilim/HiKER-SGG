import numpy as np
import torch
from torch.cuda import current_device
from torch.nn import Module, Linear
import torch.nn.functional as fn
from pickle import load as pickle_load

from model.refactor.hier import hierarchical_pred_reasoning
from model.util import MLP
from config import FCG_NODES, FCG_EDGES
from model.feature.fcg_builder import FCGBuilder
CUDA_DEVICE = torch.device(f'cuda:{current_device()}')

def wrap(nparr):
    return torch.tensor(nparr, dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)

def arange(num):
    return torch.arange(num, dtype=torch.int64, device=CUDA_DEVICE)

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

        # 初始化消息传递所需的 MLP 层
        self.fc_mp_send = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_receive = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)

        # 初始化 GRU 规则所需的线性层
        self.fc_eq3_w = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u = Linear(hidden_dim, hidden_dim)

        # 初始化输出投影层
        self.fc_output_proj = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

    def forward(self, rel_inds, obj_probs, obj_fmaps, vr):
        """
        FCG Net 的前向传播
        :param rel_inds: shape(img_all_rels,2) <s,o> 二元组
        :param obj_probs: shape(img_gt_boxes,151) boxes 的类别概率分布
        :param obj_fmaps: shape(img_gt_boxes,1024) boxes 的特征图
        :param vr: shape(img_all_rels,1024) 关系的视觉特征
        :return: pred_cls_score: 谓词的预测概率
                scpred_cls_score: 超类谓词的预测概率
        """
        num_img_gt_boxes = obj_probs.size(0)  # 图片中的实体数量
        num_img_all_rels = rel_inds.size(0)  # 图片中的关系数量

        # 复制 FCG 节点特征
        fcg_l1_nodes = self.fcg.l1_nodes
        fcg_l2_nodes = self.fcg.l2_nodes
        fcg_l3_nodes = self.fcg.l3_nodes

        # 复制 FCG 边权重
        fcg_edges_l2_l3 = self.fcg.edges_l2_l3
        fcg_edges_l1_l2 = self.fcg.edges_l1_l2  
        
        # 使用 VR 特征来作为三元组节点的特征，与 FCG 图一级节点建立桥边
        bridge_edges = torch.zeros((num_img_all_rels, len(fcg_l1_nodes)), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)

        return pred_cls_score, scpred_cls_score 