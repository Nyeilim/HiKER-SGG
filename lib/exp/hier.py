import torch
import torch.nn.functional as fn
import numpy as np

CUDA_DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

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


def hierarchical_pred_reasoning(pred_cls_logits, use_sa):
    index = torch.zeros(60 + 8, device=CUDA_DEVICE, dtype=torch.bool)
    index[0] = True
    index[51:60] = True

    scpred_cls_score = fn.softmax(pred_cls_logits[:, index], dim=1)  # img_all_rels 对空关系和 9 个一级父级谓词的预测分数
    superon_cls_score = fn.softmax(pred_cls_logits[:, 60:63], dim=1)  # img_all_rels 对二级父级谓词 superon1/2/3 的预测分数
    superof_cls_score = fn.softmax(pred_cls_logits[:, 63:66], dim=1)  # img_all_rels 对二级父级谓词 superof1/2/3 的预测分数
    superto_cls_score = fn.softmax(pred_cls_logits[:, 66:68], dim=1)  # img_all_rels 对二级父级谓词 superto1/2 的预测分数

    pred_cls_logits = pred_cls_logits[:, :51]  # 包含初始 51 个谓词【包含空关系】的相似度矩阵
    # 这段代码就是概率转移 adaptive refinement，又称为 SA(Semantic Adjustment)，使用混淆矩阵来进行概率转移
    if use_sa:
        pred_adj_np = np.load('/output/data/misc/conf_mat_updated.npy')  # 加载混淆矩阵，shape(51,51)，每行和为 1
        pred_adj_nor = torch.tensor(pred_adj_np, dtype=torch.float32, device=CUDA_DEVICE)
        pred_cls_logits = (pred_adj_nor @ pred_cls_logits.T).T  # 利用混淆矩阵实现概率转移，shape(img_all_rels, 51)

    scpred_score = torch.zeros_like(pred_cls_logits, requires_grad=True, device=CUDA_DEVICE, dtype=torch.float32)
    scpred2_score = torch.ones_like(pred_cls_logits, requires_grad=True, device=CUDA_DEVICE, dtype=torch.float32)

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
    doing_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    wear_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superon_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superat_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    position_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superin_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superof_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superto_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superother_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superon1_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superon2_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superon3_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superof1_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superof2_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superof3_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superto1_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)
    superto2_index = torch.zeros(pred_cls_logits.shape[1], requires_grad=False, device=CUDA_DEVICE, dtype=torch.bool)

    # 这里的流程和上面类似，前面算出了每个 img_all_rels 对应的空关系和一级父级谓词概率，这里把每个概率填入对应的子类格子中。
    for j in doing:
        scpred_score.data[:, j] = scpred_cls_score[:, 1]
        doing_index[j] = True  # 将列表的值转化为长度 51 的索引数组
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
    scpred_score.data[:, 0] = scpred_cls_score[:, 0]  # 把空关系的预测概率给补上

    # 区别与把整个行向量拿去做 Softmax ，这里只把归属于相同一级父级谓词的子谓词 logit 拿去做 Softmax，比如 doing = [14, 37, 47, 38]，
    # 那就把这 4 个子谓词的预测 logit 拿出来去做 Softmax，这样就有 torch.sum(pred_cls_logits[0][doing_index]) == 1
    pred_cls_logits = pred_cls_logits.type(torch.float32)
    pred_cls_logits[:, doing_index] = fn.softmax(pred_cls_logits[:, doing_index], dim=1).type(torch.float32)
    pred_cls_logits[:, wear_index] = fn.softmax(pred_cls_logits[:, wear_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superon1_index] = fn.softmax(pred_cls_logits[:, superon1_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superon2_index] = fn.softmax(pred_cls_logits[:, superon2_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superon3_index] = fn.softmax(pred_cls_logits[:, superon3_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superat_index] = fn.softmax(pred_cls_logits[:, superat_index], dim=1).type(torch.float32)
    pred_cls_logits[:, position_index] = fn.softmax(pred_cls_logits[:, position_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superin_index] = fn.softmax(pred_cls_logits[:, superin_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superof1_index] = fn.softmax(pred_cls_logits[:, superof1_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superof2_index] = fn.softmax(pred_cls_logits[:, superof2_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superof3_index] = fn.softmax(pred_cls_logits[:, superof3_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superto1_index] = fn.softmax(pred_cls_logits[:, superto1_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superto2_index] = fn.softmax(pred_cls_logits[:, superto2_index], dim=1).type(torch.float32)
    pred_cls_logits[:, superother_index] = fn.softmax(pred_cls_logits[:, superother_index], dim=1).type(torch.float32)
    pred_cls_logits[:, 0] = 1  # 空关系即是个子谓词，也是个一级父级谓词

    # 看到这里终于看懂了，rels 实际上会计算 68 个父子谓词的 logits，然后按照树状层级分别应用 Softmax 形成条件概率，用条件概率得出全局概率
    # img_all_rels[i] 属于某个子谓词的概率 = 属于某个一级父级谓词的概率 * 属于某个二级父级谓词的概率 * 在属于某父级谓词的条件下，属于某个子谓词的概率
    pred_cls_logits = pred_cls_logits * scpred_score.data * scpred2_score.data  # 逐元素乘积
    # 其实就是 18 个一级/二级父级谓词的预测分数，横向拼接在一起，shape(img_all_rels, 18)
    scpred_cls_score = torch.cat((scpred_cls_score, superon_cls_score, superof_cls_score, superto_cls_score), dim=1)

    return pred_cls_logits, scpred_cls_score