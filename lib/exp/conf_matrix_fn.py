import numpy as np
from torch import no_grad as torch_no_grad
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import BOX_SCALE, IM_SCALE
from dataloaders.visual_genome import VG, VGDataLoader
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, eval_entry
from lib.exp.global_var import conf, model, ind_to_predicates


# VG 类继承自 Dataset 类，把数据集拆分为训练集、验证集、测试集，参数作为关键字参数传入
# take train_full for evaluating the confusion matrix; return size: 57723, 5000, 26446
train_full, _val, _test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                                    use_proposals=conf.use_proposals,
                                    filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=False,
                                    get_state=False)

# VGDataLoader 类继承自 Dataloader 类，作为迭代器拿取 batch; return size: 7215, 57723
_, train_full_loader = VGDataLoader.splits(train_full, train_full, mode='rel',
                                           batch_size=conf.batch_size,
                                           num_workers=conf.num_workers,
                                           num_gpus=conf.num_gpus,
                                           pin_memory=True)

def train_evaluate(verbose = False):
    model.eval()
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes()  # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    # 该函数接收一个可迭代对象，返回一个行为与原对象相同的迭代器，但在每次请求值时打印动态更新的进度条。
    prog_bar = tqdm(enumerate(train_full_loader), total=int(len(train_full) / train_full_loader.batch_size), disable = not verbose)

    with torch_no_grad():
        for train_full_b, batch in prog_bar:
            train_full_batch(conf.num_gpus * train_full_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                             evaluator_multiple_preds_list)
            if train_full_b == 10000:  # For efficiency, only evaluate 10000 batches
                break
    confusion_matrix = evaluator[conf.mode].result_dict['predicate_confusion_matrix']
    model.train()
    return confusion_matrix


def train_full_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    with autocast():
        det_res = model[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': train_full.gt_classes[batch_num + i].copy(),
            'gt_relations': train_full.relationships[batch_num + i].copy(),
            'gt_boxes': train_full.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)
