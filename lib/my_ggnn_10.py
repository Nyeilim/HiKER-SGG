##################################################################
# From my_ggnn_09: Dynamically connecting entities to ontology too
# Also a minor change: img2ont edges are now normalized over ont rather than img
##################################################################
from pickle import load as pickle_load

import numpy as np
import torch
from torch import tensor as torch_tensor, float32 as torch_float32, \
    int64 as torch_int64, arange as torch_arange, mm as torch_mm, \
    zeros as torch_zeros, bool as torch_bool, \
    sigmoid as torch_sigmoid, tanh as torch_tanh, cat as torch_cat, zeros_like as torch_zeros_like, \
    ones_like as torch_ones_like
from torch.cuda import current_device
from torch.nn import Module, Linear, ModuleList, Sequential, ReLU, LayerNorm
from torch.nn.functional import softmax as F_softmax, relu as F_relu, \
    normalize as F_normalize

from lib.lrga import LowRankAttention
from lib.my_util import MLP, adj_normalize

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')


def wrap(nparr):
    return torch_tensor(nparr, dtype=torch_float32, device=CUDA_DEVICE, requires_grad=False)

def arange(num):
    return torch_arange(num, dtype=torch_int64, device=CUDA_DEVICE)

class GGNN(Module):
    def __init__(self, emb_path, graph_path, time_step_num=3, hidden_dim=512,
                 output_dim=512, use_embedding=True, use_knowledge=True,
                 refine_obj_cls=False, num_ents=151, num_preds=51,
                 config=None, with_clean_classifier=None, with_transfer=None,
                 num_obj_cls=None, num_rel_cls=None, sa=None, lrga=None):
        super(GGNN, self).__init__()
        self.time_step_num = time_step_num

        self.with_clean_classifier = with_clean_classifier
        self.with_transfer = with_transfer
        self.use_lrga = config.MODEL.LRGA.USE_LRGA
        self.k = config.MODEL.LRGA.K
        self.dropout = config.MODEL.LRGA.DROPOUT
        self.in_channels = hidden_dim
        self.hidden_channels = hidden_dim
        self.out_channels = hidden_dim

        self.sa = sa
        self.use_ontological_adjustment = config.MODEL.USE_ONTOLOGICAL_ADJUSTMENT
        self.normalize_eoa = config.MODEL.NORMALIZE_EOA
        self.shift_eoa = config.MODEL.SHIFT_EOA
        self.fold_eoa = config.MODEL.FOLD_EOA
        self.merge_eoa_sa = config.MODEL.MERGE_EOA_SA

        if self.use_lrga is True:
            self.attention = ModuleList()
            self.dimension_reduce = ModuleList()
            self.attention.append(LowRankAttention(self.k, self.in_channels, self.dropout))
            self.dimension_reduce.append(Sequential(Linear(2*self.k + self.hidden_channels, self.hidden_channels, device=CUDA_DEVICE), ReLU()))
            for _ in range(self.time_step_num):
                self.attention.append(LowRankAttention(self.k, self.hidden_channels, self.dropout))
                self.dimension_reduce.append(Sequential(Linear(2*self.k + self.hidden_channels, self.hidden_channels, device=CUDA_DEVICE)))
            self.dimension_reduce[-1] = Sequential(Linear(2*self.k + self.hidden_channels, self.out_channels, device=CUDA_DEVICE))
            # self.gn = ModuleList([GroupNorm(self.num_groups, self.hidden_channels) for _ in range(self.time_step_num-1)])
            self.gn = ModuleList([LayerNorm(self.hidden_channels) for _ in range(self.time_step_num-1)])

        if use_embedding: # True
            with open(emb_path, 'rb') as fin:
                self.emb_ent, self.emb_pred = pickle_load(fin)
            self.emb_ent = wrap(self.emb_ent)
            self.emb_pred = wrap(self.emb_pred)
        else:
            self.emb_ent = torch.eye(num_ents, dtype=torch_float32)
            self.emb_pred = torch.eye(num_preds, dtype=torch_float32)

        self.num_ont_ent = self.emb_ent.size(0)
        assert self.num_ont_ent == num_obj_cls + 12
        self.num_ont_pred = self.emb_pred.size(0)
        assert self.num_ont_pred == num_rel_cls + 9 + 8

        if use_knowledge: # True 是否使用知识库的边信息
            with open(graph_path, 'rb') as fin:
                edge_dict = pickle_load(fin)
            self.adjmtx_ent2ent = edge_dict['edges_ent2ent']
            self.adjmtx_ent2pred = edge_dict['edges_ent2pred']
            self.adjmtx_pred2ent = edge_dict['edges_pred2ent']
            self.adjmtx_pred2pred = edge_dict['edges_pred2pred']
        else:
            self.adjmtx_ent2ent = np.zeros((1, num_ents, num_ents), dtype=np.float32)
            self.adjmtx_ent2pred = np.zeros((1, num_ents, num_preds), dtype=np.float32)
            self.adjmtx_pred2ent = np.zeros((1, num_preds, num_ents), dtype=np.float32)
            self.adjmtx_pred2pred = np.zeros((1, num_preds, num_preds), dtype=np.float32)

        self.edges_ont_ent2ent = wrap(self.adjmtx_ent2ent)
        self.edges_ont_ent2pred = wrap(self.adjmtx_ent2pred)
        self.edges_ont_pred2ent = wrap(self.adjmtx_pred2ent)
        self.edges_ont_pred2pred = wrap(self.adjmtx_pred2pred)

        self.num_edge_types_ent2ent = self.adjmtx_ent2ent.shape[0]
        self.num_edge_types_ent2pred = self.adjmtx_ent2pred.shape[0]
        self.num_edge_types_pred2ent = self.adjmtx_pred2ent.shape[0]
        self.num_edge_types_pred2pred = self.adjmtx_pred2pred.shape[0]

        self.fc_init_ont_ent = Linear(self.emb_ent.size(1), hidden_dim)
        self.fc_init_ont_pred = Linear(self.emb_pred.size(1), hidden_dim)

        self.fc_mp_send_ont_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_ont_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)

        self.fc_mp_receive_ont_ent = MLP([(self.num_edge_types_ent2ent + self.num_edge_types_pred2ent + 1) * hidden_dim // 4,
                                          (self.num_edge_types_ent2ent + self.num_edge_types_pred2ent + 1) * hidden_dim // 4,
                                          hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_ont_pred = MLP([(self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4,
                                           (self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4,
                                           hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_ent = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_pred = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)

        self.fc_eq3_w_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_ent = Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_pred = Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_ent = Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_pred = Linear(hidden_dim, hidden_dim)

        self.fc_output_proj_img_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        self.fc_output_proj_ont_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

        self.refine_obj_cls = refine_obj_cls
        if self.refine_obj_cls:
            self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

        self.debug_info = {}

        if self.use_ontological_adjustment is True:
            print('my_ggnn_10: using use_ontological_adjustment')
            ontological_preds = self.adjmtx_pred2pred[3, :, :]
            if self.fold_eoa is True:
                diag_indices = np.diag_indices(ontological_preds.shape[0])
                folded = ontological_preds + ontological_preds.T
                folded[diag_indices] = ontological_preds[diag_indices]
            if self.shift_eoa is True:
                ontological_preds += 1.0
                print(f'EOA-N: Used shift_eoa')
            else:
                print(f'EOA-N: Not using shift_eoa. self.eoa_n={self.normalize_eoa}')
            if not self.normalize_eoa:
                ontological_preds = ontological_preds / (ontological_preds.sum(-1)[:, None] + 1e-8)
                print(f'EOA-N: Not using normalize_eoa. Using BPL\'s original normalization')
            self.ontological_preds = torch_tensor(ontological_preds, dtype=torch_float32, device=CUDA_DEVICE)
            if self.normalize_eoa is True:
                F_normalize(self.ontological_preds, out=self.ontological_preds)
                print(f'EOA-N: Used normalize_eoa')
        else:
            print(f'my_ggnn_10: not using use_ontological_adjustment. self.use_ontological_adjustment={self.use_ontological_adjustment}')

        # Init 阶段创建独属于 BPL 方法的 MLP 层以及加载混淆矩阵
        # 如果使用 BPL 方法，就会使用在这里初始化的 fc_output_proj_img_pred_clean 作为分类头，而不是 fc_output_proj_img_pred
        if self.with_clean_classifier:
            self.fc_output_proj_img_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

            if self.refine_obj_cls:
                self.fc_output_proj_img_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
                self.fc_output_proj_ont_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

            # 下面这段代码其实没啥用，你这个 self.pred_adj_nor 最后都没赋值给有效的局部变量，实际上混淆矩阵的预加载是在 global_var.py:116
            if self.with_transfer is True:
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                # 加载初始的谓词混淆矩阵
                pred_adj_np = np.load(config.MODEL.CONF_MAT_FREQ_TRAIN)
                # pred_adj_np = 1.0 - pred_adj_np
                pred_adj_np[0, :] = 0.0
                pred_adj_np[:, 0] = 0.0
                pred_adj_np[0, 0] = 1.0
                # adj_i_j means the baseline outputs category j, but the ground truth is i.
                pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)   # 行归一化，避免除零错误加了 1e-8
                if self.sa is True:
                    pred_adj_np = adj_normalize(pred_adj_np)    # 加上单位矩阵，再进行行归一化操作
                    print(f'SA: Used adj_normalize')
                else:
                    print(f'No SA: Not using adj_normalize.self.sa={self.sa}')
                self.pred_adj_nor = torch_tensor(pred_adj_np, dtype=torch_float32, device=CUDA_DEVICE)  # 转换为张量


    def forward(self, rel_inds, obj_probs, obj_fmaps, vr):
        # This is a per_image representation, not an embedding.
        num_img_ent = obj_probs.size(0) # img_gt_boxes
        num_img_pred = rel_inds.size(0) # img_all_rels = img_gt_boxes * (img_gt_boxes - 1); -1 是去除实体的自关系

        debug_info = self.debug_info
        debug_info['rel_inds'] = rel_inds
        debug_info['obj_probs'] = obj_probs

        refine_obj_cls = self.refine_obj_cls # False
        # 词嵌入来自 emb_mtx_with_sccluster2_pred_ent.pkl，包含超类节点
        nodes_ont_ent = self.fc_init_ont_ent(self.emb_ent) # 初始化 CE 节点，163(151+12) 个
        nodes_ont_pred = self.fc_init_ont_pred(self.emb_pred) # 初始化 CP 节点，68(51+9+8) 个，CP 好像有两级超类，即下面说的 Superclass, Sub-superclass
        nodes_img_ent = obj_fmaps # 初始化 SE 节点
        if self.use_lrga is True:
            original_vr = vr.clone()
        nodes_img_pred = vr # 初始化 SP 节点
        # SGG 图上的边
        edges_img_pred2subj = torch_zeros((num_img_pred, num_img_ent), dtype=torch_float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img_pred2subj[arange(num_img_pred), rel_inds[:, 0]] = 1 # 使用这个矩阵，对于某个特定的 SP 节点【行】，我们可以找到其 CE Subject【列】
        edges_img_pred2obj = torch_zeros((num_img_pred, num_img_ent), dtype=torch_float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img_pred2obj[arange(num_img_pred), rel_inds[:, 1]] = 1 # 使用这个矩阵，对于某个特定的 SP 节点【行】，我们可以找到其 CE Object【列】
        edges_img_subj2pred = edges_img_pred2subj.t()
        edges_img_obj2pred = edges_img_pred2obj.t()
        # Bridge Edge 桥边，下面这三行是 SE/CE 之间的桥边
        edges_img2ont_ent = torch_zeros((num_img_ent, self.num_ont_ent), dtype=torch_float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img2ont_ent[:, :151] = obj_probs.clone().detach() # 使用独热编码的标注，作为邻接矩阵的值；使用该矩阵，对于某个特定的 SE 节点，我们可以找到其 CE 节点
        edges_ont2img_ent = edges_img2ont_ent.t()
        ## SP/CP 之间的桥边
        edges_img2ont_pred = torch_zeros((num_img_pred, self.num_ont_pred), dtype=torch_float32, device=CUDA_DEVICE, requires_grad=False) # SP/CP 的邻接矩阵未进行初始化；使用该矩阵，对于某个特定的 SP 节点，我们可以找到其 CP 节点
        edges_ont2img_pred = edges_img2ont_pred.t()

        ent_cls_logits = None
        scent_cls_score = None
        # KG 图上的边，信息来自 all_edges_with_sccluster2_pred_ent.pkl；第一维代表着边类型 type，猜测和超类节点有关？
        edges_ont_ent2ent = self.edges_ont_ent2ent # shape(11,163,163)
        edges_ont_pred2ent = self.edges_ont_pred2ent # shape(3,68,163)
        edges_ont_ent2pred = self.edges_ont_ent2pred # shape(3,163,68)
        edges_ont_pred2pred = self.edges_ont_pred2pred # shape(6,68,68)

        num_edge_types_ent2ent = self.num_edge_types_ent2ent # 11
        num_edge_types_pred2ent = self.num_edge_types_pred2ent # 3
        num_edge_types_ent2pred = self.num_edge_types_ent2pred # 3
        num_edge_types_pred2pred = self.num_edge_types_pred2pred # 6

        with_clean_classifier = self.with_clean_classifier
        with_transfer = self.with_transfer

        # 这行代码没用啊？
        if with_clean_classifier and with_transfer:
            pred_adj_nor = self.pred_adj_nor

        for t in range(self.time_step_num): # 会进行 3 个时间步
            message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent)
            message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred)
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent)
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred)

            # NOTE: there's some vectorization opportunity right here.
            message_received_ont_ent = self.fc_mp_receive_ont_ent(torch_cat( # shape(163,1024)
                [torch_mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2ent)] + # 收集来自相连 CE 节点的信息
                [torch_mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2ent)] +  # 收集来自相连 CP 节点的信息
                [torch_mm(edges_ont2img_ent, message_send_img_ent),] # 收集来自相连 SE 节点的信息
            , 1)) # dim==1 就是在列方向上拼接，本来有个长度为 n 的列表，每个元素是个 shape(x,y) 的 tensor，拼接后将会是个 (x,y*n) 的 tensor

            message_received_ont_pred = self.fc_mp_receive_ont_pred(
                torch_cat(
                [torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)] +
                [torch_mm(edges_ont_pred2pred[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2pred)] +
                [torch_mm(edges_ont2img_pred, message_send_img_pred),]
            , 1))

            message_received_img_ent = self.fc_mp_receive_img_ent(torch_cat([
                torch_mm(edges_img_subj2pred, message_send_img_pred),
                torch_mm(edges_img_obj2pred, message_send_img_pred),
                torch_mm(edges_img2ont_ent, message_send_ont_ent),
            ], 1))

            del message_send_ont_ent, message_send_img_pred

            message_received_img_pred = self.fc_mp_receive_img_pred(torch_cat([
                torch_mm(edges_img_pred2subj, message_send_img_ent),
                torch_mm(edges_img_pred2obj, message_send_img_ent),
                torch_mm(edges_img2ont_pred, message_send_ont_pred),
            ], 1))
            # 上面这段就是信息传递的过程，最后四种节点的维度都是 1024
            del message_send_ont_pred, message_send_img_ent
            # 下面这段就是 GRU Rules
            z_ont_ent = torch_sigmoid(self.fc_eq3_w_ont_ent(message_received_ont_ent) + self.fc_eq3_u_ont_ent(nodes_ont_ent)) # 更新门
            r_ont_ent = torch_sigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent) + self.fc_eq4_u_ont_ent(nodes_ont_ent)) # 重置门
            h_ont_ent = torch_tanh(self.fc_eq5_w_ont_ent(message_received_ont_ent) + self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent)) # 候选隐状态
            del message_received_ont_ent, r_ont_ent
            # nodes_ont_ent_new = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent
            nodes_ont_ent = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent
            del z_ont_ent, h_ont_ent

            z_ont_pred = torch_sigmoid(self.fc_eq3_w_ont_pred(message_received_ont_pred) + self.fc_eq3_u_ont_pred(nodes_ont_pred))
            r_ont_pred = torch_sigmoid(self.fc_eq4_w_ont_pred(message_received_ont_pred) + self.fc_eq4_u_ont_pred(nodes_ont_pred))
            h_ont_pred = torch_tanh(self.fc_eq5_w_ont_pred(message_received_ont_pred) + self.fc_eq5_u_ont_pred(r_ont_pred * nodes_ont_pred))
            del message_received_ont_pred, r_ont_pred
            nodes_ont_pred = (1 - z_ont_pred) * nodes_ont_pred + z_ont_pred * h_ont_pred
            del z_ont_pred, h_ont_pred

            z_img_ent = torch_sigmoid(self.fc_eq3_w_img_ent(message_received_img_ent) + self.fc_eq3_u_img_ent(nodes_img_ent))
            r_img_ent = torch_sigmoid(self.fc_eq4_w_img_ent(message_received_img_ent) + self.fc_eq4_u_img_ent(nodes_img_ent))
            h_img_ent = torch_tanh(self.fc_eq5_w_img_ent(message_received_img_ent) + self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent))
            del message_received_img_ent, r_img_ent
            nodes_img_ent = (1 - z_img_ent) * nodes_img_ent + z_img_ent * h_img_ent
            del z_img_ent, h_img_ent

            z_img_pred = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred))
            r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
            h_img_pred = torch_tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
            del message_received_img_pred, r_img_pred
            nodes_img_pred = (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred
            del z_img_pred, h_img_pred
            if self.use_lrga is True: # False
                nodes_img_pred = self.dimension_reduce[t](torch_cat((self.attention[t](original_vr), nodes_img_pred), dim=1))
                if t != self.time_step_num - 1:
                    # No ReLU nor batchnorm for last layer
                    nodes_img_pred = self.gn[t](F_relu(nodes_img_pred))

            # Superclass predicate 这三个 CP 的超类没有被使用啊，实际使用的是下面两级超类 9+8=17 个超类
            geometric = [1, 2, 3, 4, 5, 8, 10, 22, 23, 28, 29, 31, 32, 33, 43]
            possesive = [6, 7, 9, 16, 17, 20, 30, 36, 27, 50, 42]
            semantic = [11, 12, 13, 14, 15, 18, 19, 21, 24, 25, 26, 34, 35, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49]

            # Superclass predicate，一级父级谓词，包含 50 个子谓词
            doing = [14, 37, 47, 38]
            wear = [48, 49]
            superon = [28, 34, 35, 26, 24, 40, 41, 31, 18]
            superat = [29, 25, 6]
            position = [10, 33, 8, 4, 2, 13]
            superin = [15, 22, 12, 45, 46]
            superof = [16, 5, 50, 23, 32, 27, 36, 30]
            superto = [1, 7, 42, 9, 19, 17, 44]
            superother = [3, 11, 20, 21, 39, 43]

            # Sub-superclass predicate，二级父级谓词，对一级父级谓词 superon, superof, superto 的再次细分，包含 24 个子谓词
            superon1 = [28, 34, 35, 18]
            superon2 = [26, 24, 40, 41]
            superon3 = [31]
            superof1 = [16, 5, 50]
            superof2 = [23, 32]
            superof3 = [27, 36, 30]
            superto1 = [1, 7, 42, 9]
            superto2 = [19, 17, 44]

            # Superclass entity
            part = [3, 40, 43, 44, 46, 58, 59, 61, 74, 82, 83, 84, 127, 129, 130, 6, 144, 57, 85]
            artifact = [4, 15, 17, 18, 19, 25, 34, 42, 50, 54, 62, 71, 75, 77, 88, 92, 97, 99, 100, 101, 102, 107, 132, 146, 148, 10, 140, 30, 47, 69, 72, 116, 117, 118, 123, 125]
            person = [20, 29, 53, 56, 68, 70, 78, 79, 90, 91, 149, 98, 119]
            clothes = [16, 31, 55, 60, 66, 67, 87, 111, 112, 113, 120, 122, 128]
            vehicle = [1, 11, 14, 23, 26, 80, 95, 135, 137, 142]
            flora = [21, 48, 51, 73, 96, 141]
            location = [7, 81, 114, 124, 131, 143, 121]
            furniture = [9, 28, 32, 36, 38, 39, 93, 108, 110, 126, 35]
            animal = [2, 8, 27, 33, 37, 41, 52, 64, 89, 109, 150, 12]
            structure = [13, 45, 63, 76, 103, 104, 105, 115, 133, 134, 136, 138, 139, 145, 147]
            building = [22, 24, 65, 106]
            food = [5, 49, 86, 94]

            # 在最后的时间步计算完毕后，开始计算全局概率计算和 SA 处理
            if t == self.time_step_num - 1:
                # 计算模
                norm_img_pred = torch.norm(nodes_img_pred, dim=1, keepdim=True)
                norm_ont_pred = torch.norm(nodes_ont_pred, dim=1, keepdim=True)

                # 归一化
                nodes_img_pred_normalized = nodes_img_pred / norm_img_pred
                nodes_ont_pred_normalized = nodes_ont_pred / norm_ont_pred

                # 是否使用全新的 MLP 层作为最后的分类头，还是说使用来自 GB-Net 的分类头？
                if with_clean_classifier:
                    # (i,j) 的值其实是两个 SP/CP 节点向量的内积，可当作相似度矩阵【但是它们模不等于1啊？】
                    pred_cls_logits = torch_mm(self.fc_output_proj_img_pred_clean(nodes_img_pred_normalized),
                                               self.fc_output_proj_ont_pred_clean(nodes_ont_pred_normalized).t())
                else:
                    pred_cls_logits = torch_mm(self.fc_output_proj_img_pred(nodes_img_pred_normalized),
                                               self.fc_output_proj_ont_pred(nodes_ont_pred_normalized).t())

                index = torch_zeros(60 + 8, requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                index[0] = True
                index[51] = True
                index[52] = True
                index[53] = True
                index[54] = True
                index[55] = True
                index[56] = True
                index[57] = True
                index[58] = True
                index[59] = True

                scpred_cls_score = F_softmax(pred_cls_logits[:, index], dim=1) # img_all_rels 对空关系和 9 个一级父级谓词的预测分数
                superon_cls_score = F_softmax(pred_cls_logits[:, 60:63], dim=1) # img_all_rels 对二级父级谓词 superon1/2/3 的预测分数
                superof_cls_score = F_softmax(pred_cls_logits[:, 63:66], dim=1) # img_all_rels 对二级父级谓词 superof1/2/3 的预测分数
                superto_cls_score = F_softmax(pred_cls_logits[:, 66:68], dim=1) # img_all_rels 对二级父级谓词 superto1/2 的预测分数
                pred_cls_logits = pred_cls_logits[:, :51] # 包含初始 51 个谓词【包含空关系】的相似度矩阵

                # 这段代码非常重要，好像就是概率转移 adaptive refinement，又称为 SA(Semantic Adjustment)，使用混淆矩阵来进行概率转移
                # 然后概率转移之后 pred_cls_logits 每行的概率之和不等于 1，所以需要归一化，应该就是下面的操作
                if self.with_transfer:
                    pred_adj_np = np.load('/output/data/misc/conf_mat_updated.npy')  # 加载混淆矩阵
                    pred_adj_nor = torch_tensor(pred_adj_np, dtype=torch_float32, device=CUDA_DEVICE) # shape(51,51), torch.sum(pred_adj_nor, dim=1) == [1,1,1,1...]
                    pred_cls_logits = (pred_adj_nor @ pred_cls_logits.T).T # 利用混淆矩阵实现概率转移，shape(img_all_rels, 51)

                scpred_score = torch_zeros_like(pred_cls_logits, requires_grad=True, device=CUDA_DEVICE, dtype=torch_float32)
                scpred2_score = torch_ones_like(pred_cls_logits, requires_grad=True, device=CUDA_DEVICE, dtype=torch_float32)

                # scpred2_score.shape(img_all_rels, 51)，前面算出了每个 img_all_rels 对应的二级父级谓词概率，这里把每个概率填入对应的子类格子中。
                # 比如 img_all_rels[0] 属于二级父级谓词 superon1 的概率为 0.3，
                # 而 scpred2_score[0][28] 这个子类谓词 28 又属于二级父级谓词 superon1，则 scpred2_score[0][28] = 0.3
                for i in superon1:
                    scpred2_score.data[:, i] = superon_cls_score[:, 0]
                for i in superon2:
                    scpred2_score.data[:, i] = superon_cls_score[:, 1]
                for i in superon3:
                    scpred2_score.data[:, i] = superon_cls_score[:, 2]
                for i in superof1:
                    scpred2_score.data[:, i] = superof_cls_score[:, 0]
                for i in superof2:
                    scpred2_score.data[:, i] = superof_cls_score[:, 1]
                for i in superof3:
                    scpred2_score.data[:, i] = superof_cls_score[:, 2]
                for i in superto1:
                    scpred2_score.data[:, i] = superto_cls_score[:, 0]
                for i in superto2:
                    scpred2_score.data[:, i] = superto_cls_score[:, 1]

                # 下面都是长度为 51 的张量，作为索引数组
                doing_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                wear_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superon_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superat_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                position_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superin_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superof_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superto_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superother_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superon1_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superon2_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superon3_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superof1_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superof2_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superof3_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superto1_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                superto2_index = torch_zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)

                # 这里的流程和上面类似，前面算出了每个 img_all_rels 对应的空关系和一级父级谓词概率，这里把每个概率填入对应的子类格子中。
                for j in doing:
                    scpred_score.data[:, j] = scpred_cls_score[:, 1]
                    doing_index[j] = True # 将列表的值转化为长度 51 的索引数组
                for j in wear:
                    scpred_score.data[:, j] = scpred_cls_score[:, 2]
                    wear_index[j] = True
                for j in superon:
                    scpred_score.data[:, j] = scpred_cls_score[:, 3]
                    superon_index[j] = True
                for j in superat:
                    scpred_score.data[:, j] = scpred_cls_score[:, 4]
                    superat_index[j] = True
                for j in position:
                    scpred_score.data[:, j] = scpred_cls_score[:, 5]
                    position_index[j] = True
                for j in superin:
                    scpred_score.data[:, j] = scpred_cls_score[:, 6]
                    superin_index[j] = True
                for j in superof:
                    scpred_score.data[:, j] = scpred_cls_score[:, 7]
                    superof_index[j] = True
                for j in superto:
                    scpred_score.data[:, j] = scpred_cls_score[:, 8]
                    superto_index[j] = True
                for j in superother:
                    scpred_score.data[:, j] = scpred_cls_score[:, 9]
                    superother_index[j] = True
                # 这里是上边忘了创建二级父级谓词的索引数组，这里补上
                for j in superon1:
                    superon1_index[j] = True
                for j in superon2:
                    superon2_index[j] = True
                for j in superon3:
                    superon3_index[j] = True
                for j in superof1:
                    superof1_index[j] = True
                for j in superof2:
                    superof2_index[j] = True
                for j in superof3:
                    superof3_index[j] = True
                for j in superto1:
                    superto1_index[j] = True
                for j in superto2:
                    superto2_index[j] = True
                scpred_score.data[:, 0] = scpred_cls_score[:, 0] # 把空关系的预测概率给补上

                # 区别与把整个行向量拿去做 Softmax ，这里只把归属于相同一级父级谓词的子谓词 logit 拿去做 Softmax，比如 doing = [14, 37, 47, 38]，
                # 那就把这 4 个子谓词的预测 logit 拿出来去做 Softmax，这样就有 torch.sum(pred_cls_logits[0][doing_index]) == 1
                pred_cls_logits = pred_cls_logits.type(torch_float32)
                pred_cls_logits[:, doing_index] = F_softmax(pred_cls_logits[:, doing_index], dim=1).type(torch_float32)
                pred_cls_logits[:, wear_index] = F_softmax(pred_cls_logits[:, wear_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superon1_index] = F_softmax(pred_cls_logits[:, superon1_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superon2_index] = F_softmax(pred_cls_logits[:, superon2_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superon3_index] = F_softmax(pred_cls_logits[:, superon3_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superat_index] = F_softmax(pred_cls_logits[:, superat_index], dim=1).type(torch_float32)
                pred_cls_logits[:, position_index] = F_softmax(pred_cls_logits[:, position_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superin_index] = F_softmax(pred_cls_logits[:, superin_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superof1_index] = F_softmax(pred_cls_logits[:, superof1_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superof2_index] = F_softmax(pred_cls_logits[:, superof2_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superof3_index] = F_softmax(pred_cls_logits[:, superof3_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superto1_index] = F_softmax(pred_cls_logits[:, superto1_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superto2_index] = F_softmax(pred_cls_logits[:, superto2_index], dim=1).type(torch_float32)
                pred_cls_logits[:, superother_index] = F_softmax(pred_cls_logits[:, superother_index], dim=1).type(torch_float32)
                pred_cls_logits[:, 0] = 1 # 空关系即是个子谓词，也是个一级父级谓词
                # 看到这里终于看懂了，rels 实际上会计算 68 个父子谓词的 logits，然后按照树状层级分别应用 Softmax 形成条件概率，用条件概率得出全局概率
                # img_all_rels[i] 属于某个子谓词的概率 = 属于某个一级父级谓词的概率 * 属于某个二级父级谓词的概率 * 在属于某父级谓词的条件下，属于某个子谓词的概率
                pred_cls_logits = pred_cls_logits * scpred_score.data * scpred2_score.data # 逐元素乘积

                # 其实就是 18 个一级/二级父级谓词的预测分数，横向拼接在一起，shape(img_all_rels, 18)
                scpred_cls_score = torch_cat((scpred_cls_score, superon_cls_score, superof_cls_score, superto_cls_score), dim=1)

            # -----------------------
            # 上面是使用 BPL 方法的逻辑

            edges_img2ont_pred = F_softmax(pred_cls_logits, dim=1) # 前面那个是 SP/CP 相似度矩阵，这边通过 Softmax 将其压缩到 (0,1)
            edges_ont2img_pred = edges_img2ont_pred.t()
            if refine_obj_cls: # False
                ent_cls_logits = torch_mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = F_softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()
                if t == self.time_step_num - 1:
                    index = torch_zeros(163, requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)

                    index[0] = True
                    index[151] = True
                    index[152] = True
                    index[153] = True
                    index[154] = True
                    index[155] = True
                    index[156] = True
                    index[157] = True
                    index[158] = True
                    index[159] = True
                    index[160] = True
                    index[161] = True
                    index[162] = True

                    scent_cls_score = F_softmax(ent_cls_logits[:, index], dim=1)
                    ent_cls_logits = ent_cls_logits[:, :151]

                    scent_score = torch_zeros_like(ent_cls_logits, requires_grad=True, device=CUDA_DEVICE, dtype=torch_float32)

                    part_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    artifact_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    person_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    clothes_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    vehicle_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    flora_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    location_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    furniture_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    animal_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    structure_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    building_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)
                    food_index = torch_zeros(ent_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch_bool)

                    for j in part:
                        scent_score.data[:, j] = scent_cls_score[:, 1]
                        part_index[j] = True
                    for j in artifact:
                        scent_score.data[:, j] = scent_cls_score[:, 2]
                        artifact_index[j] = True
                    for j in person:
                        scent_score.data[:, j] = scent_cls_score[:, 3]
                        person_index[j] = True
                    for j in clothes:
                        scent_score.data[:, j] = scent_cls_score[:, 4]
                        clothes_index[j] = True
                    for j in vehicle:
                        scent_score.data[:, j] = scent_cls_score[:, 5]
                        vehicle_index[j] = True
                    for j in flora:
                        scent_score.data[:, j] = scent_cls_score[:, 6]
                        flora_index[j] = True
                    for j in location:
                        scent_score.data[:, j] = scent_cls_score[:, 7]
                        location_index[j] = True
                    for j in furniture:
                        scent_score.data[:, j] = scent_cls_score[:, 8]
                        furniture_index[j] = True
                    for j in animal:
                        scent_score.data[:, j] = scent_cls_score[:, 9]
                        animal_index[j] = True
                    for j in structure:
                        scent_score.data[:, j] = scent_cls_score[:, 10]
                        structure_index[j] = True
                    for j in building:
                        scent_score.data[:, j] = scent_cls_score[:, 11]
                        building_index[j] = True
                    for j in food:
                        scent_score.data[:, j] = scent_cls_score[:, 12]
                        food_index[j] = True
                    scent_score.data[:, 0] = scent_cls_score[:, 0]

                    ent_cls_logits = ent_cls_logits.type(torch_float32)
                    ent_cls_logits[:, part_index] = F_softmax(ent_cls_logits[:, part_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, artifact_index] = F_softmax(ent_cls_logits[:, artifact_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, person_index] = F_softmax(ent_cls_logits[:, person_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, clothes_index] = F_softmax(ent_cls_logits[:, clothes_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, vehicle_index] = F_softmax(ent_cls_logits[:, vehicle_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, flora_index] = F_softmax(ent_cls_logits[:, flora_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, location_index] = F_softmax(ent_cls_logits[:, location_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, furniture_index] = F_softmax(ent_cls_logits[:, furniture_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, animal_index] = F_softmax(ent_cls_logits[:, animal_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, structure_index] = F_softmax(ent_cls_logits[:, structure_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, building_index] = F_softmax(ent_cls_logits[:, building_index], dim=1).type(torch_float32)
                    ent_cls_logits[:, food_index] = F_softmax(ent_cls_logits[:, food_index], dim=1).type(torch_float32)

                    ent_cls_logits[:, 0] = 1
                    ent_cls_logits = ent_cls_logits * scent_score.data

        return pred_cls_logits, ent_cls_logits, scpred_cls_score, scent_cls_score
