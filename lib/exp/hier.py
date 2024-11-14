import numpy as np
import torch
import torch.nn.functional as fn

CUDA_DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# Superclass predicate 这三个 CP 的超类没有被使用啊，实际使用的是下面两级超类 9+8=17 个超类
# geometric = [1, 2, 3, 4, 5, 8, 10, 22, 23, 28, 29, 31, 32, 33, 43]
# possesive = [6, 7, 9, 16, 17, 20, 30, 36, 27, 50, 42]
# semantic = [11, 12, 13, 14, 15, 18, 19, 21, 24, 25, 26, 34, 35, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49]

# 层级分类信息
# Superclass predicate，一级超类谓词，包含 50 个子谓词
doing = [14, 37, 47, 38]
wear = [48, 49]
superon = [28, 34, 35, 26, 24, 40, 41, 31, 18]
superat = [29, 25, 6]
position = [10, 33, 8, 4, 2, 13]
superin = [15, 22, 12, 45, 46]
superof = [16, 5, 50, 23, 32, 27, 36, 30]
superto = [1, 7, 42, 9, 19, 17, 44]
superother = [3, 11, 20, 21, 39, 43]

# Sub-superclass predicate，二级超类谓词，对一级超类谓词 superon, superof, superto 的再次细分，包含 24 个子谓词
superon1 = [28, 34, 35, 18]
superon2 = [26, 24, 40, 41]
superon3 = [31]
superof1 = [16, 5, 50]
superof2 = [23, 32]
superof3 = [27, 36, 30]
superto1 = [1, 7, 42, 9]
superto2 = [19, 17, 44]

# 特殊谓词 non-relationship，下面将其归类到一级超类谓词
non = [0]

# 整合谓词超类信息
scpred_list = [non, doing, wear, superon, superat, position, superin, superof, superto, superother]  # 10 个
scpred2_list = [superon1, superon2, superon3, superof1, superof2, superof3, superto1, superto2]  # 8 个
smallest_scpred_list = [non, doing, wear, superon1, superon2, superon3, superat,
                        position, superin, superof1, superof2, superof3, superto1, superto2,
                        superother]  # 最小超类，最小超类的叶子为子谓词

# Superclass entity，超类实体
part = [3, 40, 43, 44, 46, 58, 59, 61, 74, 82, 83, 84, 127, 129, 130, 6, 144, 57, 85]
artifact = [4, 15, 17, 18, 19, 25, 34, 42, 50, 54, 62, 71, 75, 77, 88, 92, 97, 99, 100, 101, 102, 107, 132, 146, 148,
            10, 140, 30, 47, 69, 72, 116, 117, 118, 123, 125]
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

# 特殊超类 _background
background = [0]

# 整合实体超类信息
scent_list = [background, part, artifact, person, clothes, vehicle, flora, location,
              furniture, animal, structure, building, food]  # 13 个

# pred_cls_logits.shape(img_all_rels, 68)
def hierarchical_pred_reasoning(pred_cls_logits, use_sa):
    scpred_index = [0, *range(51, 60)]
    scpred_cls_score = fn.softmax(pred_cls_logits[:, scpred_index], dim=1)  # 每个关系对 10 个一级超类谓词【包含空关系】的预测分数汇总，总和为 1
    superon_cls_score = fn.softmax(pred_cls_logits[:, 60:63], dim=1)  # 每个关系对二级父级谓词 superon1/2/3 的预测分数
    superof_cls_score = fn.softmax(pred_cls_logits[:, 63:66], dim=1)  # 每个关系对二级父级谓词 superof1/2/3 的预测分数
    superto_cls_score = fn.softmax(pred_cls_logits[:, 66:68], dim=1)  # 每个关系对二级父级谓词 superto1/2 的预测分数
    scpred2_cls_score = torch.cat((superon_cls_score, superof_cls_score, superto_cls_score), dim=1)  # 每个关系对 8 个二级超类谓词的预测分数汇总，总和为 3
    pred_cls_logits = pred_cls_logits[:, :51]  # 包含初始 51 个谓词【包含空关系】的相似度信息的矩阵

    # 这段代码就是概率转移 adaptive refinement，又称为 SA(Semantic Adjustment)，使用混淆矩阵来进行概率转移
    if use_sa:
        pred_adj_np = np.load('/output/data/misc/conf_mat_updated.npy')  # 加载混淆矩阵，shape(51,51)，每行和为 1
        pred_adj_nor = torch.tensor(pred_adj_np, dtype=torch.float32, device=CUDA_DEVICE)
        pred_cls_logits = (pred_adj_nor @ pred_cls_logits.T).T  # 利用混淆矩阵实现概率转移，shape(img_all_rels, 51)

    # 每个子谓词一定有一级超类谓词，但不一定有二级超类谓词，因此缺省的二级超类谓词用 1 填充
    scpred_score = torch.zeros_like(pred_cls_logits, device=CUDA_DEVICE, dtype=torch.float32)
    scpred2_score = torch.ones_like(pred_cls_logits, device=CUDA_DEVICE, dtype=torch.float32)

    # scpred2_score.shape(img_all_rels, 51)，前面算出了每个关系对应的二级超类谓词概率 scpred2_cls_score，这里把概率填入对应的子谓词格子中。
    # 比如某关系属于二级谓词 superon1 的概率为 0.3，那么它的子类 [28, 34, 35, 18] 对应的格子都要填上 0.3
    for i, scpred2 in enumerate(scpred2_list):  # scpred2_list 和 scpred2_cls_score 的谓词排序一致，因此索引号 i 可以共用
        for j in scpred2:  # 遍历从属于二级超类谓词 scpred2 的子谓词
            scpred2_score[:, j] = scpred2_cls_score[:, i]  # 把二级超类谓词 scpred2 的概率填入子谓词的格子里

    # 流程和上面类似，前面算出了每个关系对应的一级超类谓词概率 scpred_cls_score，这里把概率填入对应的子类格子中。
    for i, scpred in enumerate(scpred_list):
        for j in scpred:
            scpred_score[:, j] = scpred_cls_score[:, i]

    # 区别与把整个行向量拿去做 Softmax ，这里只把归属于相同一级超类谓词的子谓词 logit 拿去做 Softmax，比如 doing = [14, 37, 47, 38]，
    # 那就把这 4 个子谓词的预测 logit 拿出来去做 Softmax，这样就有 torch.sum(pred_cls_logits[:,doing]) == [1,1,...]
    pred_cls_logits = pred_cls_logits.type(torch.float32)
    pred_cls_cond_score = torch.zeros_like(pred_cls_logits, device=CUDA_DEVICE, dtype=torch.float32)  # 条件概率

    # 计算属于某最小超类的条件下，属于某个子谓词的条件概率
    for smallest_scpred in smallest_scpred_list:
        pred_cls_cond_score[:, smallest_scpred] = fn.softmax(pred_cls_logits[:, smallest_scpred], dim=1).type(
            torch.float32)

    # 每个关系实际上会计算 68 个父子谓词的 logits，然后按照树状层级分别应用 Softmax 形成条件概率，用条件概率得出全局概率
    # 某个关系属于某个子谓词的概率 = 属于某个一级超类谓词的概率 * 属于某个二级超类谓词的概率 * 属于某最小超类的条件下，属于某个子谓词的条件概率
    pred_cls_score = scpred_score * scpred2_score * pred_cls_cond_score  # 逐元素乘积

    # 将 18 个一级/二级超类谓词的预测分数横向拼接在一起，shape(img_all_rels, 18)
    scpred_cls_score = torch.cat((scpred_cls_score, scpred2_cls_score), dim=1)

    return pred_cls_score, scpred_cls_score


def hierarchical_ent_reasoning(ent_cls_logits):
    scent_index = [0, *range(151, 163)]
    scent_cls_score = fn.softmax(ent_cls_logits[:, scent_index], dim=1)
    ent_cls_logits = ent_cls_logits[:, :151]

    # 填充超类概率矩阵
    scent_score = torch.zeros_like(ent_cls_logits, device=CUDA_DEVICE, dtype=torch.float32)
    for i, scent in enumerate(scent_list):
        for j in scent:
            scent_score[:, j] = scent_cls_score[:, i]

    # 填充条件概率矩阵
    scent_cls_cond_score = torch.zeros_like(ent_cls_logits, device=CUDA_DEVICE, dtype=torch.float32)
    ent_cls_logits = ent_cls_logits.type(torch.float32)
    smallest_scent_list = scent_list  # 因为实体超类的分类只有一级，所以直接就是最小超类
    for smallest_scent in smallest_scent_list:
        scent_cls_cond_score[:, smallest_scent] = fn.softmax(ent_cls_logits[:, smallest_scent], dim=1).type(
            torch.float32)

    # 全局概率
    ent_cls_score = scent_score * scent_cls_cond_score

    return ent_cls_score, scent_cls_score
