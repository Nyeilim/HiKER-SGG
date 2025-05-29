import torch
import torch.nn.functional as fn
from model.feature.fcg_builder import FCGBuilder
from torch.cuda import current_device
from config import logger

CUDA_DEVICE = torch.device(f'cuda:{current_device()}')


def post_process(pred_cls_score, normal_rel_mask):
    """
    后处理函数，将 pred_cls_score 扩充到原来的大小
    """
    # 创建一个全零矩阵，大小为 (normal_rel_mask.size(0), 51)
    expanded_pred_cls_score = torch.zeros((normal_rel_mask.size(0), 51), dtype=torch.float32, device=CUDA_DEVICE)
    # 将第一列(0号列)全部设为1
    expanded_pred_cls_score[:, 0] = 1
    # 将原始的 pred_cls_score 填充到对应位置
    expanded_pred_cls_score[normal_rel_mask] = pred_cls_score
    # 更新 pred_cls_score
    pred_cls_score = expanded_pred_cls_score

    return pred_cls_score


def hierarchical_reasoning_fcg(fcg: FCGBuilder, bridge_edges_tri_l1, triplet, fcg_l2_feats, fcg_l3_feats):

    assert bridge_edges_tri_l1.size(0) == triplet.size(0)
    num_img_all_rels = bridge_edges_tri_l1.size(0)
    num_l2_nodes = fcg_l2_feats.size(0)
    num_l3_nodes = fcg_l3_feats.size(0)

    # 构筑 l1->l2 l1->l3 l2->l3 的子节点层级下标映射
    l1_l3_idx_map_sub = {}
    l1_l3_idx_map_obj = {}
    l1_l2_idx_map = {}
    l2_l3_idx_map = {}

    for l1_node in fcg.l1_nodes:
        l2_subnodes = fcg.find_subnode(l1_node)
        l2_subnodes_idx = [l2_node.idx for l2_node in l2_subnodes]
        l1_l2_idx_map[l1_node.idx] = l2_subnodes_idx

        l3_subnodes = []
        for l2_node in l2_subnodes:
            l3_subnodes.extend(fcg.find_subnode(l2_node))
        l3_subnodes_idx = [l3_node.idx for l3_node in l3_subnodes]

        if l1_node.sub is not None:
            l1_l3_idx_map_sub[l1_node.idx] = l3_subnodes_idx
        if l1_node.obj is not None:
            l1_l3_idx_map_obj[l1_node.idx] = l3_subnodes_idx

    for l2_node in fcg.l2_nodes:
        l3_subnodes = fcg.find_subnode(l2_node)
        l3_subnodes_idx = [l3_node.idx for l3_node in l3_subnodes]
        l2_l3_idx_map[l2_node.idx] = l3_subnodes_idx

    l2_hier_prob_logit = torch.mm(triplet, fcg_l2_feats.t())
    l3_hier_prob_logit = torch.mm(triplet, fcg_l3_feats.t())

    # 计算 l2/l3 的层次（条件）概率
    l1_hier_prob = bridge_edges_tri_l1
    l2_hier_prob = torch.zeros((num_img_all_rels, num_l2_nodes), dtype=torch.float32, device=CUDA_DEVICE)
    l3_hier_prob = torch.zeros((num_img_all_rels, num_l3_nodes), dtype=torch.float32, device=CUDA_DEVICE)

    for l1_idx, l2_subnodes_idx in l1_l2_idx_map.items():
        l2_hier_prob[:, l2_subnodes_idx] = fn.softmax(l2_hier_prob_logit[:, l2_subnodes_idx], dim=1,
                                                      dtype=torch.float32)

    for l2_idx, l3_subnodes_idx in l2_l3_idx_map.items():
        l3_hier_prob[:, l3_subnodes_idx] = fn.softmax(l3_hier_prob_logit[:, l3_subnodes_idx], dim=1,
                                                      dtype=torch.float32)

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
        l2_hier_cpt_matrix[:, l3_subnodes_idx] = l2_hier_prob[:, l2_idx].unsqueeze(1)

    total_cls_prob = (l1_hier_cpt_matrix_sub * l2_hier_cpt_matrix * l3_hier_cpt_matrix) + (
            l1_hier_cpt_matrix_obj * l2_hier_cpt_matrix * l3_hier_cpt_matrix)

    # 合并三元组概率，反映射回谓词概率
    pred_cls_prob = torch.zeros((num_img_all_rels, 51), dtype=torch.float32, device=CUDA_DEVICE)
    for i, l3_node in enumerate(fcg.l3_nodes):
        pred_cls_prob[:, l3_node.pred] += total_cls_prob[:, i]

    return pred_cls_prob


def pred_center_reasoning(pred_center, triplet):
    """
    比较 pred_center 和 triplet 之间的相似度，并返回分类结果
    :param pred_center: 形状为(num_pred_centers, hidden_dim) 的谓词中心特征
    :param triplet: 形状为(num_relations, hidden_dim) 的三元组特征
    :return: pred_cls_prob: 形状为(num_relations, num_pred_centers) 的分类概率
    """
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(triplet, pred_center.t())  # 形状为(num_relations, num_pred_centers)

    # 应用 softmax 获取概率分布
    pred_cls_prob = fn.softmax(similarity_matrix, dim=1)

    return pred_cls_prob


def pre_process(fcg: FCGBuilder, rel_inds, ent_probs, triplet):
    """
    预处理函数
    1. 构建 ent_idx -> l1_node.idx 的映射
    2. 剔除没在训练集中出现的关系
    3. 返回 bridge_edges_tri_l1, normal_rel_mask, triplet
    """
    num_img_all_rels = triplet.size(0)

    # 数据集中 idx 映射到数组中 idx
    fcg_l1_nodes = fcg.l1_nodes
    l1_nodes_sub_idx_map = {}  # ent_idx -> l1_node.idx
    l1_nodes_obj_idx_map = {}  # ent_idx -> l1_node.idx
    for i, node in enumerate(fcg_l1_nodes):
        if node.sub is not None:  # <sub,x,x>
            l1_nodes_sub_idx_map[node.sub] = i
        if node.obj is not None:  # <x,x,obj>
            l1_nodes_obj_idx_map[node.obj] = i
    # sub_mask = torch.zeros((151,))
    # sub_mask[list(l1_nodes_sub_idx_map.keys())] = 1
    # obj_mask = torch.zeros((151,))
    # obj_mask[list(l1_nodes_obj_idx_map.keys())] = 1

    # 1. 根据 rel_inds 找到关系对应的 gt_boxes
    sub_boxes = rel_inds[:, 0]  # 主语对应的 box 索引
    obj_boxes = rel_inds[:, 1]  # 宾语对应的 box 索引

    # 2. 获取 boxes 对应的实体类别概率分布
    sub_probs = ent_probs[sub_boxes]  # (num_rels, 151) 主语的类别概率
    obj_probs = ent_probs[obj_boxes]  # (num_rels, 151) 宾语的类别概率

    # 初始化 normal_rel_mask 为全 1 张量
    normal_rel_mask = torch.ones(num_img_all_rels, dtype=torch.bool, device=CUDA_DEVICE)

    # 3. 剔除 <s,o> 对不存在的样本
    count = 0
    for i in range(num_img_all_rels):
        s_max = torch.argmax(sub_probs[i]).item()  # 拿到最大概率的实体索引
        o_max = torch.argmax(obj_probs[i]).item()  # 改进点：可以考虑 Top-K
        if not fcg.has_sample(s_max, o_max):
            normal_rel_mask[i] = False
            count += 1

    # 布尔切片
    sub_probs = sub_probs[normal_rel_mask, :]
    obj_probs = obj_probs[normal_rel_mask, :]
    triplet = triplet[normal_rel_mask, :]
    num_img_all_rels_filtered = triplet.size(0)
    logger.debug('rels num in this image | filter out: {}, left: {}'.format(count, num_img_all_rels_filtered))

    # 4. 建立桥边
    # 使用 VR 特征来作为三元组节点的特征，与 FCG 图一级节点建立桥边
    bridge_edges_tri_l1 = torch.zeros((num_img_all_rels_filtered, len(fcg_l1_nodes)), dtype=torch.float32,
                                      device=CUDA_DEVICE)

    # 对于每个关系,将主语概率分配给对应的主语模式一级节点
    for i in range(num_img_all_rels_filtered):
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

    return bridge_edges_tri_l1, normal_rel_mask, triplet