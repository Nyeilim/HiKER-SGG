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

from model.refactor.hier import hierarchical_ent_reasoning, hierarchical_pred_reasoning
from lib.kern_old.lrga import LowRankAttention
from model.util import MLP, adj_normalize
from config import CONF_MAT_FREQ_TRAIN, MODEL, EDGE_MATRIX, PRED_BRIDGE_EDGE_INITIAL
from model.feature.bridge_prior import ContextAwarePrior

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
        self.use_lrga = MODEL.LRGA.USE_LRGA
        self.k = MODEL.LRGA.K
        self.dropout = MODEL.LRGA.DROPOUT
        self.in_channels = hidden_dim
        self.hidden_channels = hidden_dim
        self.out_channels = hidden_dim

        self.sa = sa
        self.use_ontological_adjustment = MODEL.USE_ONTOLOGICAL_ADJUSTMENT
        self.normalize_eoa = MODEL.NORMALIZE_EOA
        self.shift_eoa = MODEL.SHIFT_EOA
        self.fold_eoa = MODEL.FOLD_EOA
        self.merge_eoa_sa = MODEL.MERGE_EOA_SA

        # 新增属性
        self.normalize_classifier = False
        self.pred_bridge_edge_initial = PRED_BRIDGE_EDGE_INITIAL
        print("normalize_classifier: {} @ {}".format(self.normalize_classifier, __name__))

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

        # Init 阶段创建独属于 BPL 方法的 MLP 层以及加载混淆矩阵
        # 如果使用 BPL 方法，就会使用在这里初始化的 fc_output_proj_img_pred_clean 作为分类头，而不是 fc_output_proj_img_pred
        if self.with_clean_classifier:
            self.fc_output_proj_img_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

            if self.refine_obj_cls:
                self.fc_output_proj_img_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
                self.fc_output_proj_ont_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

            # 下面这段代码其实没啥用，你这个 self.pred_adj_nor 最后都没赋值给有效的局部变量，实际上混淆矩阵的预加载是在 hikersgg_predcls_train.py:55
            if self.with_transfer is True:
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                # 加载初始的谓词混淆矩阵
                pred_adj_np = np.load(CONF_MAT_FREQ_TRAIN)
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

        # 新增上下文感知的桥边初始化器
        self.context_prior = None  # 延迟初始化，等待edge_matrix

        # ============== DPL 组件初始化 ==============
        self.use_dpl = config.MODEL.USE_DPL if hasattr(config.MODEL, 'USE_DPL') else False

        if self.use_dpl:
            print(f'[GGNN] Enabling DPL with N_DIM={config.MODEL.DPL.N_DIM}')

            # DPL 特征压缩层
            self.dpl_dim = config.MODEL.DPL.N_DIM  # 默认 128
            self.rel_compress_dpl = Linear(hidden_dim, self.dpl_dim)

            # 原型嵌入 (每个谓词类一个原型)
            self.proto_emb = torch.nn.Parameter(
                torch.zeros(num_rel_cls, self.dpl_dim),
                requires_grad=True
            )
            torch.nn.init.orthogonal_(self.proto_emb)

            # 高斯参数网络
            self.gaussian_emb = Linear(self.dpl_dim, self.dpl_dim * 2)

            # 距离转换参数
            self.shift = torch.nn.Parameter(torch.ones(1) * 15.0)
            self.negative_scale = torch.nn.Parameter(torch.ones(1) * 15.0)

            # DPL 超参数
            self.dpl_sample_size = config.MODEL.DPL.AVG_NUM_SAMPLE
            self.dpl_alpha = config.MODEL.DPL.ALPHA
            self.dpl_radius = config.MODEL.DPL.RADIUS

            # 采样数量数组（可选：根据类别频率调整）
            if config.MODEL.DPL.FREQ_BASED_DIFF_N:
                # TODO: 需要提供 VG 数据集的谓词计数文件
                print('[DPL] FREQ_BASED_DIFF_N enabled but not implemented yet, using uniform sampling')
                self.sample_size_array = np.ones(num_rel_cls, dtype=int) * self.dpl_sample_size
            else:
                self.sample_size_array = np.ones(num_rel_cls, dtype=int) * self.dpl_sample_size

            # 测试时配置
            self.dpl_use_in_test = config.MODEL.DPL.USE_IN_TEST
            self.dpl_fusion_weight = config.MODEL.DPL.FUSION_WEIGHT

    def forward(self, rel_inds, obj_probs, obj_fmaps, vr, rel_labels=None):
        """
        GGNN Rules 内核
        :param rel_inds: shape(img_all_rels,2) <s,o> 二元组
        :param obj_probs: shape(img_gt_boxes,151)
        :param obj_fmaps: gt_boxes 所在区域的特征图 shape(img_gt_boxes,1024)
        :param vr: rel 的视觉特征 shape(img_all_rels,1024)
        :param rel_labels: shape(img_all_rels) 关系标签（仅训练时提供）
        :return: 谓词的预测概率 pred_cls_score 超类谓词的预测概率 scpred_cls_score
        """
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
        assert torch.all((0 <= rel_inds[:, 0]) & (rel_inds[:, 0] < num_img_ent))
        assert torch.all((0 <= rel_inds[:, 1]) & (rel_inds[:, 1] < num_img_ent))
        edges_img_pred2subj = torch.zeros((num_img_pred, num_img_ent), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img_pred2obj = torch.zeros((num_img_pred, num_img_ent), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img_pred2subj[arange(num_img_pred), rel_inds[:, 0]] = 1 # 使用这个矩阵，对于某个特定的 SP 节点【行】，我们可以找到其 CE Subject【列】
        edges_img_pred2obj[arange(num_img_pred), rel_inds[:, 1]] = 1 # 使用这个矩阵，对于某个特定的 SP 节点【行】，我们可以找到其 CE Object【列】
        edges_img_subj2pred = edges_img_pred2subj.t()
        edges_img_obj2pred = edges_img_pred2obj.t()

        # Bridge Edge 桥边
        ## SE/CE 之间的桥边，使用独热编码的标注，作为邻接矩阵的值；使用该矩阵，对于某个特定的 SE 节点，我们可以找到其 CE 节点
        edges_img2ont_ent = torch.zeros((num_img_ent, self.num_ont_ent), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        edges_img2ont_ent[:, :151] = obj_probs.clone().detach()
        edges_ont2img_ent = edges_img2ont_ent.t()
        ## SP/CP 之间的桥边，SP/CP 的邻接矩阵未进行初始化；使用该矩阵，对于某个特定的 SP 节点，我们可以找到其 CP 节点
        edges_img2ont_pred = torch.zeros((num_img_pred, self.num_ont_pred), dtype=torch.float32, device=CUDA_DEVICE, requires_grad=False)
        if self.pred_bridge_edge_initial:
            if self.context_prior is None:
                edge_matrix = np.load(EDGE_MATRIX)
                self.context_prior = ContextAwarePrior(edge_matrix)
            edges_img2ont_pred[:, :51] = self.context_prior.get_context_aware_prior(rel_inds, obj_probs.argmax(1))
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

            # 消息传递循环结束后，使用 HA 层对齐向量空间
            # nodes_img_pred, nodes_ont_pred = self.double_ha(nodes_img_pred.unsqueeze(1), nodes_ont_pred.unsqueeze(1))
            # nodes_img_pred = nodes_img_pred.squeeze(1)
            # nodes_ont_pred = nodes_ont_pred.squeeze(1)

            # 是否使用全新的MLP层作为最后的分类头
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
            edges_img2ont_pred = fn.softmax(pred_cls_logits, dim=1) # 通过 Softmax 将其压缩到 (0,1)，更新桥边连接权重
            edges_ont2img_pred = edges_img2ont_pred.t()

            if refine_obj_cls: # False
                ent_cls_logits = torch.mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = fn.softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()

        # ============== DPL 分支 ==============
        dpl_logits = None
        dpl_losses = {}

        if self.use_dpl:
            # 压缩特征到 DPL 维度
            dpl_features = self.rel_compress_dpl(nodes_img_pred)  # (num_img_pred, dpl_dim)

            # 计算 DPL logits 和损失
            dpl_logits, dpl_losses = self._compute_dpl(
                dpl_features,
                rel_labels=rel_labels  # 从 forward 参数传入
            )

        # 循环结束，进行最后的层级分类
        pred_cls_score, scpred_cls_score = hierarchical_pred_reasoning(pred_cls_logits, with_transfer)
        if refine_obj_cls:  # False
            ent_cls_score, scent_cls_score = hierarchical_ent_reasoning(ent_cls_logits)

        # 修改返回值，包含 DPL logits 和损失
        if self.use_dpl:
            return pred_cls_score, ent_cls_score, scpred_cls_score, scent_cls_score, dpl_logits, dpl_losses
        else:
            return pred_cls_score, ent_cls_score, scpred_cls_score, scent_cls_score, None, {}

    def _compute_dpl(self, features, rel_labels=None):
        """
        计算 DPL 的 logits 和损失

        Args:
            features: (num_rel, dpl_dim) 压缩后的关系特征
            rel_labels: (num_rel,) 关系标签，仅训练时提供

        Returns:
            dpl_logits: (num_rel, num_rel_cls) DPL 分类 logits
            add_losses: dict, 包含 'ortho_loss' 和 'sample_loss'
        """
        add_losses = {}

        # 归一化原型
        predicate_proto = self.proto_emb
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)

        # 计算高斯参数
        gaussian = self.gaussian_emb(predicate_proto_norm)
        mu, logsigma = torch.split(gaussian, self.dpl_dim, dim=1)

        # 计算特征到原型的距离
        num_rel = features.size(0)
        rel_rep_expand = features.unsqueeze(1).expand(-1, self.proto_emb.size(0), -1)
        proto_expand = predicate_proto_norm.unsqueeze(0).expand(num_rel, -1, -1)
        distance_set = (rel_rep_expand - proto_expand).norm(dim=2)

        # 距离转换为 logits
        dpl_logits = -self.negative_scale * distance_set + self.shift

        # ========== 训练阶段：计算额外损失 ==========
        if self.training and rel_labels is not None:
            # 1. 正交损失：确保不同类别原型相互正交
            proto_sim = torch.matmul(predicate_proto_norm, predicate_proto_norm.t())
            ortho_loss = self._get_orth_loss(proto_sim)
            add_losses['ortho_loss'] = ortho_loss

            # 2. 采样损失：从高斯分布采样，强制特征落在原型半径内
            detach_proto = predicate_proto_norm.detach()
            z = self._sample_gaussian_tensors(
                detach_proto, logsigma, self.sample_size_array
            ).view(-1, self.dpl_dim)

            # 计算到采样点的距离
            distance_set_z = self._distance(features, z)

            # 找到每个类别的最小距离
            distance_set_m = self._get_min_dists_z(distance_set_z, self.sample_size_array)

            # 选择正确类别的距离
            selected_distance = distance_set_m[torch.arange(rel_labels.size(0)), rel_labels]

            # Hinge loss: 距离应该在半径内
            zeros_tensor = torch.zeros_like(selected_distance)
            sample_loss = torch.mean(
                torch.where(
                    selected_distance > self.dpl_radius,
                    torch.pow(selected_distance - self.dpl_radius, 2),
                    zeros_tensor
                )
            )
            add_losses['sample_loss'] = sample_loss * self.dpl_alpha

        # ========== 测试阶段：使用方差加权距离 ==========
        elif not self.training:
            nd = ((rel_rep_expand - proto_expand) / (logsigma.exp() + 1e-8)).norm(dim=2)
            ndn = (nd.t() / (nd.max(dim=1)[0] + 1e-8) * distance_set.max(dim=1)[0]).t()
            dpl_logits = -self.negative_scale * ndn + self.shift

        return dpl_logits, add_losses

    def _get_orth_loss(self, proto_sim):
        """计算原型正交损失"""
        eye_sim = torch.triu(torch.ones_like(proto_sim), diagonal=1)
        loss_orth = torch.abs(proto_sim[eye_sim == 1]).mean()
        return loss_orth

    def _sample_gaussian_tensors(self, mu, logsigma, num_samples):
        """从高斯分布采样"""
        total_samples = self._sample_each(mu[0], logsigma[0], num_samples[0])
        for i in range(1, mu.size(0)):
            samples = self._sample_each(mu[i], logsigma[i], num_samples[i])
            total_samples = torch.cat([total_samples, samples], dim=0)
        return total_samples

    def _sample_each(self, mu_i, logsigma_i, num_samples_i):
        """为单个类别采样"""
        eps = torch.randn(num_samples_i, mu_i.size(0), dtype=mu_i.dtype, device=mu_i.device)
        samples = eps.mul(torch.exp(logsigma_i)).add_(mu_i)
        return samples

    def _distance(self, t1, t2):
        """计算欧氏距离矩阵"""
        t1_square = torch.sum(t1 ** 2, dim=1, keepdim=True)
        t2_square = torch.sum(t2 ** 2, dim=1)
        distance_set = torch.sqrt(
            t1_square + t2_square - 2 * torch.matmul(t1, t2.t()) + 1e-8
        )
        return distance_set

    def _get_min_dists_z(self, rel_dists_z, num_samples):
        """获取到采样点的最小距离"""
        total_md = rel_dists_z[:, :num_samples[0]].sort(1, descending=False)[0][:, :1]
        count = num_samples[0]
        for i in range(1, len(num_samples)):
            md = rel_dists_z[:, count:count+num_samples[i]].sort(1, descending=False)[0][:, :1]
            total_md = torch.cat([total_md, md], dim=1)
            count = count + num_samples[i]
        return total_md
