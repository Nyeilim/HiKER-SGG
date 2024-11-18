##################################################################
# From my_ggnn_09: Dynamically connecting entities to ontology too
# Also a minor change: img2ont edges are now normalized over ont rather than img
##################################################################
from pickle import load as pickle_load

import numpy as np
import torch
from torch.cuda import current_device
from torch.nn import Module, Linear, ModuleList, Sequential, ReLU, LayerNorm
import torch.nn.functional as fn

from lib.exp.hier import hierarchical_ent_reasoning, hierarchical_pred_reasoning
from lib.lrga import LowRankAttention
from lib.my_util import MLP, adj_normalize

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')


def wrap(nparr):
    return torch.tensor(nparr, dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)

def arange(num):
    return torch.arange(num, dtype=torch.int64, device=CUDA_DEVICE)

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

        self.normalize_classifier = False

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
            self.emb_ent = torch.eye(num_ents, dtype=torch.float32)
            self.emb_pred = torch.eye(num_preds, dtype=torch.float32)

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
            self.ontological_preds = torch.tensor(ontological_preds, dtype=torch.float32, device=CUDA_DEVICE)
            if self.normalize_eoa is True:
                fn.normalize(self.ontological_preds, out=self.ontological_preds)
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
                self.pred_adj_nor = torch.tensor(pred_adj_np, dtype=torch.float32, device=CUDA_DEVICE)  # 转换为张量


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
        nodes_img_pred = vr  # 初始化 SP 节点

        original_vr = None
        if self.use_lrga is True:
            original_vr = vr.clone()

        # SGG 图上的边
        assert torch.all((0 <= rel_inds[:, 0]) & (rel_inds[:, 0] <= 2))
        assert torch.all((0 <= rel_inds[:, 0]) & (rel_inds[:, 0] <= 2))
        edges_img_pred2subj = torch.zeros((num_img_pred, num_img_ent), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img_pred2subj[arange(num_img_pred), rel_inds[:, 0]] = 1 # 使用这个矩阵，对于某个特定的 SP 节点【行】，我们可以找到其 CE Subject【列】
        edges_img_pred2obj = torch.zeros((num_img_pred, num_img_ent), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img_pred2obj[arange(num_img_pred), rel_inds[:, 1]] = 1 # 使用这个矩阵，对于某个特定的 SP 节点【行】，我们可以找到其 CE Object【列】
        edges_img_subj2pred = edges_img_pred2subj.t()
        edges_img_obj2pred = edges_img_pred2obj.t()
        
        # Bridge Edge 桥边
        ## SE/CE 之间的桥边，使用独热编码的标注，作为邻接矩阵的值；使用该矩阵，对于某个特定的 SE 节点，我们可以找到其 CE 节点
        edges_img2ont_ent = torch.zeros((num_img_ent, self.num_ont_ent), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img2ont_ent[:, :151] = obj_probs.clone().detach() # 
        edges_ont2img_ent = edges_img2ont_ent.t()
        ## SP/CP 之间的桥边，SP/CP 的邻接矩阵未进行初始化；使用该矩阵，对于某个特定的 SP 节点，我们可以找到其 CP 节点
        edges_img2ont_pred = torch.zeros((num_img_pred, self.num_ont_pred), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_ont2img_pred = edges_img2ont_pred.t()

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

        # 返回值定义
        pred_cls_score = None
        ent_cls_score = None
        scpred_cls_score = None
        scent_cls_score = None
        
        # 中间变量
        pred_cls_logits = None
        ent_cls_logits = None
        
        # 这行代码没用
        if with_clean_classifier and with_transfer:
            pred_adj_nor = self.pred_adj_nor

        for t in range(self.time_step_num): # 会进行 3 个时间步
            message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent)
            message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred)
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent)
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred)

            # NOTE: there's some vectorization opportunity right here.
            message_received_ont_ent = self.fc_mp_receive_ont_ent(torch.cat( # shape(163,1024)
                [torch.mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2ent)] + # 收集来自相连 CE 节点的信息
                [torch.mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2ent)] +  # 收集来自相连 CP 节点的信息
                [torch.mm(edges_ont2img_ent, message_send_img_ent),] # 收集来自相连 SE 节点的信息
            , 1)) # dim==1 就是在列方向上拼接，本来有个长度为 n 的列表，每个元素是个 shape(x,y) 的 tensor，拼接后将会是个 (x,y*n) 的 tensor

            message_received_ont_pred = self.fc_mp_receive_ont_pred(
                torch.cat(
                [torch.mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)] +
                [torch.mm(edges_ont_pred2pred[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2pred)] +
                [torch.mm(edges_ont2img_pred, message_send_img_pred),]
            , 1))

            message_received_img_ent = self.fc_mp_receive_img_ent(torch.cat([
                torch.mm(edges_img_subj2pred, message_send_img_pred),
                torch.mm(edges_img_obj2pred, message_send_img_pred),
                torch.mm(edges_img2ont_ent, message_send_ont_ent),
            ], 1))

            del message_send_ont_ent, message_send_img_pred

            message_received_img_pred = self.fc_mp_receive_img_pred(torch.cat([
                torch.mm(edges_img_pred2subj, message_send_img_ent),
                torch.mm(edges_img_pred2obj, message_send_img_ent),
                torch.mm(edges_img2ont_pred, message_send_ont_pred),
            ], 1))
            
            del message_send_ont_pred, message_send_img_ent
            
            # 上面这段就是信息传递的过程，最后四种节点的维度都是 1024
            # ----------------------------
            # 下面这段就是 GRU Rules
            
            z_ont_ent = torch.sigmoid(self.fc_eq3_w_ont_ent(message_received_ont_ent) + self.fc_eq3_u_ont_ent(nodes_ont_ent)) # 更新门
            r_ont_ent = torch.sigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent) + self.fc_eq4_u_ont_ent(nodes_ont_ent)) # 重置门
            h_ont_ent = torch.tanh(self.fc_eq5_w_ont_ent(message_received_ont_ent) + self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent)) # 候选隐状态
            del message_received_ont_ent, r_ont_ent
            # nodes_ont_ent_new = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent
            nodes_ont_ent = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent
            del z_ont_ent, h_ont_ent

            z_ont_pred = torch.sigmoid(self.fc_eq3_w_ont_pred(message_received_ont_pred) + self.fc_eq3_u_ont_pred(nodes_ont_pred))
            r_ont_pred = torch.sigmoid(self.fc_eq4_w_ont_pred(message_received_ont_pred) + self.fc_eq4_u_ont_pred(nodes_ont_pred))
            h_ont_pred = torch.tanh(self.fc_eq5_w_ont_pred(message_received_ont_pred) + self.fc_eq5_u_ont_pred(r_ont_pred * nodes_ont_pred))
            del message_received_ont_pred, r_ont_pred
            nodes_ont_pred = (1 - z_ont_pred) * nodes_ont_pred + z_ont_pred * h_ont_pred
            del z_ont_pred, h_ont_pred

            z_img_ent = torch.sigmoid(self.fc_eq3_w_img_ent(message_received_img_ent) + self.fc_eq3_u_img_ent(nodes_img_ent))
            r_img_ent = torch.sigmoid(self.fc_eq4_w_img_ent(message_received_img_ent) + self.fc_eq4_u_img_ent(nodes_img_ent))
            h_img_ent = torch.tanh(self.fc_eq5_w_img_ent(message_received_img_ent) + self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent))
            del message_received_img_ent, r_img_ent
            nodes_img_ent = (1 - z_img_ent) * nodes_img_ent + z_img_ent * h_img_ent
            del z_img_ent, h_img_ent

            z_img_pred = torch.sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred))
            r_img_pred = torch.sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
            h_img_pred = torch.tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
            del message_received_img_pred, r_img_pred
            nodes_img_pred = (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred
            del z_img_pred, h_img_pred
            
            # ---------------------
            
            if self.use_lrga is True: # False
                nodes_img_pred = self.dimension_reduce[t](torch.cat((self.attention[t](original_vr), nodes_img_pred), dim=1))
                if t != self.time_step_num - 1:
                    # No ReLU nor batchnorm for last layer
                    nodes_img_pred = self.gn[t](fn.relu(nodes_img_pred))

            # 是否使用全新的 MLP 层作为最后的分类头，还是说使用来自 GB-Net 的分类头？
            if with_clean_classifier:
                nodes_img_pred_fc = self.fc_output_proj_img_pred_clean(nodes_img_pred)
                nodes_ont_pred_fc = self.fc_output_proj_ont_pred_clean(nodes_ont_pred)
            else:
                nodes_img_pred_fc = self.fc_output_proj_img_pred(nodes_img_pred)
                nodes_ont_pred_fc = self.fc_output_proj_ont_pred(nodes_ont_pred)

            # 计算模并归一化；不要进行归一化，他会让你的指标降 20 个点！
            if self.normalize_classifier:
                nodes_img_pred_fc = nodes_img_pred_fc / torch.norm(nodes_img_pred_fc, dim=1, keepdim=True)
                nodes_ont_pred_fc = nodes_ont_pred_fc / torch.norm(nodes_ont_pred_fc, dim=1, keepdim=True)

            # (i,j) 的值其实是两个 SP/CP 节点向量的内积，在两者尺度差别不大的情况下可当作相似度矩阵
            pred_cls_logits = torch.mm(nodes_img_pred_fc,nodes_ont_pred_fc.t())
            edges_img2ont_pred = fn.softmax(pred_cls_logits, dim=1) # 通过 Softmax 将其压缩到 (0,1)
            edges_ont2img_pred = edges_img2ont_pred.t()

            if refine_obj_cls: # False
                ent_cls_logits = torch.mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = fn.softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()

        # 循环结束，进行最后的层级分类
        pred_cls_score, scpred_cls_score = hierarchical_pred_reasoning(pred_cls_logits, with_transfer)
        if refine_obj_cls:  # False
            ent_cls_score, scent_cls_score = hierarchical_ent_reasoning(ent_cls_logits)
        
        return pred_cls_score, ent_cls_score, scpred_cls_score, scent_cls_score
