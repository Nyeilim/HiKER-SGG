import numpy as np
import torch
from torch.cuda import current_device
from torch.nn import Module, Linear
import torch.nn.functional as fn

from model.util import MLP
from model.feature.fcg_builder import FCGBuilder

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

        # 初始化消息传递所需的 MLP 层
        self.mlp_send_l1_nodes = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_send_l2_nodes = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_send_l3_nodes = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_send_triplet = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.mlp_rcv_l1_nodes = MLP([2 * hidden_dim // 4, 2 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.mlp_rcv_l2_nodes = MLP([2 * hidden_dim // 4, 2 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.mlp_rcv_l3_nodes = MLP([hidden_dim // 4, hidden_dim // 2, hidden_dim], act_fn='ReLU', last_act=True)
        self.mlp_rcv_triplet = MLP([hidden_dim // 4, hidden_dim // 2, hidden_dim], act_fn='ReLU', last_act=True)

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
        num_img_gt_boxes = ent_probs.size(0)  # 图片中的实体数量
        num_img_all_rels = rel_inds.size(0)  # 图片中的关系数量
        triplet = vr.clone()  # 使用关系视觉特征作为三元组特征

        # 复制 FCG 节点特征
        fcg_l1_feats = torch.stack([node.feat for node in self.fcg.l1_nodes]).to(CUDA_DEVICE)
        fcg_l2_feats = torch.stack([node.feat for node in self.fcg.l2_nodes]).to(CUDA_DEVICE)
        fcg_l3_feats = torch.stack([node.feat for node in self.fcg.l3_nodes]).to(CUDA_DEVICE)

        # 复制 FCG 边权重
        fcg_edges_l2_l3 = self.fcg.edges_l2_l3.to(CUDA_DEVICE)
        fcg_edges_l1_l2 = self.fcg.edges_l1_l2.to(CUDA_DEVICE)
        fcg_edges_l3_l2 = fcg_edges_l2_l3.t()
        fcg_edges_l2_l1 = fcg_edges_l1_l2.t()

        # 数据集中 idx 映射到数组中 idx
        fcg_l1_nodes = self.fcg.l1_nodes
        l1_nodes_sub_idx_map = {}
        l1_nodes_obj_idx_map = {}
        for i, node in enumerate(fcg_l1_nodes):
            if node.sub is not None:  # <sub,x,x>
                l1_nodes_sub_idx_map[node.sub] = i
            if node.obj is not None:  # <x,x,obj>
                l1_nodes_obj_idx_map[node.obj] = i
        sub_mask = torch.zeros((151,))
        sub_mask[list(l1_nodes_sub_idx_map.keys())] = 1
        obj_mask = torch.zeros((151,))
        obj_mask[list(l1_nodes_obj_idx_map.keys())] = 1

        # 使用 VR 特征来作为三元组节点的特征，与 FCG 图一级节点建立桥边
        bridge_edges_tri_l1 = torch.zeros((num_img_all_rels, len(fcg_l1_nodes)), dtype=torch.float32, device=CUDA_DEVICE)
        
        # 1. 根据 rel_inds 找到关系对应的 gt_boxes
        sub_boxes = rel_inds[:, 0]  # 主语对应的 box 索引
        obj_boxes = rel_inds[:, 1]  # 宾语对应的 box 索引

        # 2. 获取 boxes 对应的实体类别概率分布
        sub_probs = ent_probs[sub_boxes]  # (num_rels, 151) 主语的类别概率
        obj_probs = ent_probs[obj_boxes]  # (num_rels, 151) 宾语的类别概率

        # 3. 剔除不在 l1_nodes 中的实体概率并归一化
        sub_probs = sub_probs * sub_mask.to(CUDA_DEVICE)
        obj_probs = obj_probs * obj_mask.to(CUDA_DEVICE)

        # 归一化概率
        # TODO：这里归一化出错怎么办？因为有大量无效数据，他们可能根本没有在训练集中出现
        sub_probs = fn.normalize(sub_probs, p=1, dim=1)
        obj_probs = fn.normalize(obj_probs, p=1, dim=1)

        # 4. 建立桥边
        # 对于每个关系,将主语概率分配给对应的主语模式一级节点
        for i in range(num_img_all_rels):
            for ent_idx, prob in enumerate(sub_probs[i]):
                if prob > 0 and ent_idx in l1_nodes_sub_idx_map:
                    bridge_edges_tri_l1[i][l1_nodes_sub_idx_map[ent_idx]] = prob

            # 将宾语概率分配给对应的宾语模式一级节点
            for ent_idx, prob in enumerate(obj_probs[i]):
                if prob > 0 and ent_idx in l1_nodes_obj_idx_map:
                    bridge_edges_tri_l1[i][l1_nodes_obj_idx_map[ent_idx]] = prob

        # 归一化桥边权重，使每个关系的总权重为 1；这边的主宾语权重融合可以做考虑，现在是 1/2 的情况
        # 可以考虑看分布的最大概率，如果主语的最大概率高于宾语，那么预测分支应该更加偏向于主语分支
        bridge_edges_tri_l1 = fn.normalize(bridge_edges_tri_l1, p=1, dim=1)
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
                torch.mm(bridge_edges_l1_tri, msg_send_l1_nodes),
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

        pred_cls_score = self.hierarchical_reasoning_fcg(bridge_edges_tri_l1, triplet, fcg_l2_feats, fcg_l3_feats)
        return pred_cls_score

    def hierarchical_reasoning_fcg(self, bridge_edges_tri_l1, triplet, fcg_l2_feats, fcg_l3_feats):

        num_img_all_rels = bridge_edges_tri_l1.size(0)
        num_l2_nodes = fcg_l2_feats.size(0)
        num_l3_nodes = fcg_l3_feats.size(0)

        # 构筑 l1->l2 l1->l3 l2->l3 的子节点层级下标映射
        l1_l3_idx_map_sub = {}
        l1_l3_idx_map_obj = {}
        l1_l2_idx_map = {}
        l2_l3_idx_map = {}

        for l1_node in self.fcg.l1_nodes:
            l2_subnodes = self.fcg.find_subnode(l1_node)
            l2_subnodes_idx = [l2_node.idx for l2_node in l2_subnodes]
            l1_l2_idx_map[l1_node.idx] = l2_subnodes_idx

            l3_subnodes = []
            for l2_node in l2_subnodes:
                l3_subnodes.extend(self.fcg.find_subnode(l2_node))
            l3_subnodes_idx = [l3_node.idx for l3_node in l3_subnodes]

            if l1_node.sub is not None:
                l1_l3_idx_map_sub[l1_node.idx] = l3_subnodes_idx
            if l1_node.obj is not None:
                l1_l3_idx_map_obj[l1_node.idx] = l3_subnodes_idx

        for l2_node in self.fcg.l2_nodes:
            l3_subnodes = self.fcg.find_subnode(l2_node)
            l3_subnodes_idx = [l3_node.idx for l3_node in l3_subnodes]
            l2_l3_idx_map[l2_node.idx] = l3_subnodes_idx

        l2_hier_prob_logit = torch.mm(triplet, fcg_l2_feats.t())
        l3_hier_prob_logit = torch.mm(triplet, fcg_l3_feats.t())

        # 计算 l2/l3 的层次（条件）概率
        l1_hier_prob = bridge_edges_tri_l1
        l2_hier_prob = torch.zeros((num_img_all_rels, num_l2_nodes), dtype=torch.float32, device=CUDA_DEVICE)
        l3_hier_prob = torch.zeros((num_img_all_rels, num_l3_nodes), dtype=torch.float32, device=CUDA_DEVICE)

        for l1_idx, l2_subnodes_idx in l1_l2_idx_map.items():
            l2_hier_prob[:, l2_subnodes_idx] = fn.softmax(l2_hier_prob_logit[:, l2_subnodes_idx], dim=1, dtype=torch.float32)

        for l2_idx, l3_subnodes_idx in l2_l3_idx_map.items():
            l3_hier_prob[:, l3_subnodes_idx] = fn.softmax(l3_hier_prob_logit[:, l3_subnodes_idx], dim=1, dtype=torch.float32)

        # 填充计算矩阵，三个计算矩阵的尺寸为 (num_img_all_rels, num_l3_nodes)，将三个矩阵作逐元素累乘，即为分类到 l3 某节点的最终概率
        l1_hier_cpt_matrix_sub = torch.zeros((num_img_all_rels, num_l3_nodes), dtype=torch.float32, device=CUDA_DEVICE)
        l1_hier_cpt_matrix_obj = torch.zeros((num_img_all_rels, num_l3_nodes), dtype=torch.float32, device=CUDA_DEVICE)
        l2_hier_cpt_matrix = torch.zeros((num_img_all_rels, num_l3_nodes), dtype=torch.float32, device=CUDA_DEVICE)
        l3_hier_cpt_matrix = l3_hier_prob

        # 值得注意，对于某个 l3 节点，从 l1 节点出发会有来自 sub/obj 的两条路径，因此需要将两条路径的概率相加
        # 在处理上，需要将 l1_hier_cpt_matrix 拆分为两个矩阵，避免 l3_subnodes_idx 上的值被重复覆盖
        for l1_idx, l3_subnodes_idx in l1_l3_idx_map_sub.items():
            l1_hier_cpt_matrix_sub[:, l3_subnodes_idx] = l1_hier_prob[:, l1_idx].unsqueeze(1)

        for l1_idx, l3_subnodes_idx in l1_l3_idx_map_obj.items():
            l1_hier_cpt_matrix_obj[:, l3_subnodes_idx] = l1_hier_prob[:, l1_idx].unsqueeze(1)

        for l2_idx, l3_subnodes_idx in l2_l3_idx_map.items():
            l2_hier_cpt_matrix[:, l3_subnodes_idx] = l2_hier_prob[:, l2_idx]

        total_cls_prob = (l1_hier_cpt_matrix_sub * l2_hier_cpt_matrix * l3_hier_cpt_matrix) + (
                    l1_hier_cpt_matrix_obj * l2_hier_cpt_matrix * l3_hier_cpt_matrix)

        # 合并三元组概率，反映射回谓词概率
        pred_cls_prob = torch.zeros((num_img_all_rels, 51), dtype=torch.float32, device=CUDA_DEVICE)
        for i, l3_node in enumerate(self.fcg.l3_nodes):
            pred_cls_prob[:, l3_node.pred] += total_cls_prob[:, i]

        return pred_cls_prob
