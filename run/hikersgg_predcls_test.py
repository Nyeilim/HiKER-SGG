import os
import sys

import numpy as np
import torch
from torch import no_grad as torch_no_grad
from torch.cuda.amp import autocast
from tqdm import tqdm
from torchinfo import summary

sys.path.append("/output/HiKER-SGG/")

from config import ModelConfig, BOX_SCALE, IM_SCALE
from lib.exp.exp_util import load_best_matrices
from lib.pytorch_misc import optimistic_restore
# 如果把 sg_val 放在 my_model_24, visual_genome 后面就会导入报错，因为里面有个很重要的 setup 语句能导入 lib.fpn.box_intersections_cpu.bbox
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import print_para
from dataloaders.visual_genome import VGDataLoader, VG

from lib.my_model_24 import KERN

os.environ["CUDA_VISIBLE_DEVICES"]="0"
codebase = '/output/HiKER-SGG/'
exp_name = 'hikersgg_predcls_test'
write = tqdm.write  # 函数引用赋值，用来打印日志
use_bpl = True # 启用还是关闭 BPL 方法
use_sa = False # 启用还是关闭 SA 方法
# test_epoch = load_best_matrices()['best_mr_epoch'] # 需要测试第几个 epoch 训练出来的模型
test_epoch = 2
print(f"Start test epoch {test_epoch}")

# Change ckpt path for the evaluated model
# If you want to test on VG-C benchmark, include "-test_n" in the command line
conf = ModelConfig(f'''
-m predcls -p 2500 -clip 5
-ckpt /output/data/checkpoints/kern_predcls/hikersgg_predcls_train/vgrel-{test_epoch}.tar
-b 8
-ngpu 1
-nwork 24
-test
-lr 1e-4
''')

# Also load the corresponding confusion matrix, Remember to change to the path of the confusion matrix
matrix_suffix = test_epoch - (test_epoch + 1) % 3   # 混淆矩阵会每三轮计算一次
# 设置模型对应的混淆矩阵
if matrix_suffix < 2:
    conf_matrix =  np.load('/output/data/misc/conf_mat_freq_train.npy')
else:
    conf_matrix = np.load(f'/output/data/misc/conf/conf_mat_updated_{matrix_suffix}.npy')
np.save('/output/data/misc/conf_mat_updated.npy', conf_matrix)
conf.MODEL.CONF_MAT_FREQ_TRAIN = '/output/data/misc/conf_mat_freq_train.npy'
conf.MODEL.LRGA.USE_LRGA = False
conf.MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
conf.MODEL.NORMALIZE_EOA = False
# conf.MODEL.LRGA.K = 50
# conf.MODEL.LRGA.DROPOUT = 0.5
# conf.MODEL.GN.NUM_GROUPS = 1024//8

# 数据集加载
train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                            use_proposals=conf.use_proposals,
                            filter_non_overlap=conf.mode == 'sgdet',
                            with_clean_classifier=False,
                            get_state=False, test_n=conf.test_n)

ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship
# Evaluate on test set. 这里说白了调用的就是 val 方法，只不过用的 test 测试集做样本，而不是 val 验证集。
# Here we let val = test since we want to call val_epoch() for evaluation
if conf.test or conf.test_n:
    val = test  # 为了复用 val_epoch 方法，将 test 数据集赋值给 val
_, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)

# 创建模型对象
detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                graph_path=os.path.join(codebase, 'graphs/005/all_edges_with_sccluster2_pred_ent.pkl'),
                emb_path=os.path.join(codebase, 'graphs/001/emb_mtx_with_sccluster2_pred_ent.pkl'),
                rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                class_volume=1.0, with_clean_classifier=use_bpl, with_transfer=use_sa, sa=use_sa, config=conf,
               )

def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    with autocast():
        det_res = detector[b]   # 这个就是模型的入口，Blob 类型，调用 KERN __getitem__ 方法进行 batch 分发
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)

def val_epoch():
    detector.eval()
    evaluator_list = [] # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes() # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    prog_bar = tqdm(enumerate(val_loader), total=int(len(val)/val_loader.batch_size), disable=True) # 关闭进度条

    with torch_no_grad():
        for val_b, batch in prog_bar:
            val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)

    detector.train()
    return recall, recall_mp, mean_recall, mean_recall_mp

ckpt = torch.load(conf.ckpt)    # 加载参数文件
optimistic_restore(detector, ckpt['state_dict'], skip_clean=False)  # 参数导入模型中
detector.cuda() # 模型移至 CUDA
# print(print_para(detector), flush=True) # 打印模型参数
# print(detector) # 原生方法打印模型结构
one_sample = next(iter(val_loader))
one_sample.scatter()
summary(detector, input_data= [*one_sample[0]]) # 使用 torchinfo 打印模型信息
detector.eval() # 评估模式，禁用梯度记录
recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()  # 开始评估