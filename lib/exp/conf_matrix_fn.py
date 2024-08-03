import numpy as np
import torch
from torch import no_grad as torch_no_grad
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import BOX_SCALE, IM_SCALE
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, eval_entry
from lib.exp.global_var import conf, detector, ind_to_predicates, train_full_loader, train_full
from lib.my_util import adj_normalize
from lib.pytorch_misc import optimistic_restore

# Initialize the confusion matrix
initial_conf_matrix = np.load(conf.MODEL.CONF_MAT_FREQ_TRAIN)
initial_conf_matrix[0, :] = 0.0
initial_conf_matrix[:, 0] = 0.0
initial_conf_matrix[0, 0] = 1.0
initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
initial_conf_matrix = adj_normalize(initial_conf_matrix)
np.save('/output/data/misc/conf_mat_updated.npy', initial_conf_matrix)

# ckpt = torch.load(conf.ckpt)
# optimistic_restore(detector, ckpt['state_dict'], skip_clean=False)
# detector.cuda()

def train_evaluate():
    detector.eval()
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes()  # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    prog_bar = tqdm(enumerate(train_full_loader), total=int(len(train_full) / train_full_loader.batch_size))

    with torch_no_grad():
        for train_full_b, batch in prog_bar:
            train_full_batch(conf.num_gpus * train_full_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                             evaluator_multiple_preds_list)
            if train_full_b == 10000:  # For efficiency, only evaluate 10000 batches
                break
    confusion_matrix = evaluator[conf.mode].result_dict['predicate_confusion_matrix']
    detector.train()
    return confusion_matrix


def train_full_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    with autocast():
        det_res = detector[b]
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
