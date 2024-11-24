import numpy as np
from torch import no_grad as torch_no_grad
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import BOX_SCALE, IM_SCALE, DIS_PROGRESS_BAR, data_path
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mr, eval_entry


def confusion_matrix_evaluate(model, conf, matrix_val_set, matrix_val_set_loader):
    _, conf_matrix = _val_epoch(model, conf, matrix_val_set, matrix_val_set_loader, matrix_eval=True)
    return conf_matrix


def val_epoch(model, conf, val_set, val_set_loader):
    matrices, _ = _val_epoch(model, conf, val_set, val_set_loader, matrix_eval=False)
    return matrices

def val_batch(
        model, conf, val, batch_num, batch,
        evaluator, evaluator_multiple_preds,
        evaluator_list, evaluator_multiple_preds_list
):
    with autocast():
        det_res = model[batch]
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
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)


def _val_epoch(model, conf, dataset, dataloader, matrix_eval):
    ind_to_predicates = dataset.ind_to_predicates

    model.eval()
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []

    # 为每个谓词创建两个评估器：单谓词评估器、多谓词评估器
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes()  # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    # 该函数接收一个可迭代对象，返回一个行为与原对象相同的迭代器，但在每次请求值时打印动态更新的进度条。
    prog_bar = tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size), disable=DIS_PROGRESS_BAR)

    with torch_no_grad():
        for batch_idx, batch in prog_bar:
            val_batch(
                model, conf, dataset, conf.num_gpus * batch_idx, batch,
                evaluator, evaluator_multiple_preds,
                evaluator_list, evaluator_multiple_preds_list
            )
            if matrix_eval and batch_idx == 10000:  # For efficiency, only evaluate 10000 batches while matrix_eval
                break

    # confusion matrix
    confusion_matrix = evaluator[conf.mode].result_dict['predicate_confusion_matrix']
    if not matrix_eval:
        print('~~~~~~~~ Confusion Matrix in Val Epoch ~~~~~~~~')
        print(confusion_matrix)
        dump_file = data_path('confusion_matrix.npy')
        np.save(confusion_matrix, dump_file)

    # matrices; mp(multiple preds) equals `without constraint`
    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
    mean_recall = calculate_mr(evaluator_list, conf.mode)
    mean_recall_mp = calculate_mr(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)
    matrices = (recall, recall_mp, mean_recall, mean_recall_mp)

    model.train()
    return matrices, confusion_matrix
