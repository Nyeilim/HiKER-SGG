import numpy as np
from torch import no_grad as torch_no_grad
from torch.cuda.amp import autocast
from time import time as time_time
from tqdm import tqdm

from config import BOX_SCALE, IM_SCALE, DIS_PROGRESS_BAR, data_path, logger
from model.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mr, eval_entry, save_per_predicate_recall_json, save_recall_samples_json


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
    forward_start = time_time()  # 前向传播计时开始
    with autocast():
        det_res = model[batch]
    if conf.num_gpus == 1:
        det_res = [det_res]
    forward_time = time_time() - forward_start
    logger.debug(f"前向传播总耗时: {forward_time:.4f}s")

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        # 真实标注
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }

        # 添加 image_id（如果存在）
        if hasattr(val, 'img_ids') and len(val.img_ids) > batch_num + i:
            gt_entry['image_id'] = val.img_ids[batch_num + i]
        elif hasattr(val, 'img_to_first_rel') and len(val.img_to_first_rel) > batch_num + i:
            # 尝试从 img_to_first_rel 获取 image_id
            gt_entry['image_id'] = batch_num + i  # 使用索引作为 fallback
        else:
            gt_entry['image_id'] = batch_num + i  # 使用索引作为 fallback
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
        # 预测结果
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
    ind_to_classes = dataset.ind_to_classes  # 添加 ind_to_classes

    model.eval()

    # 判断是否需要跟踪召回样本（仅测试 PredCls 且 constrained recall）
    track_recall_samples = (conf.test and conf.mode == 'predcls' and not matrix_eval)

    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []

    # 为每个谓词创建两个评估器：单谓词评估器、多谓词评估器
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes(track_recall_samples=track_recall_samples, ind_to_predicates=ind_to_predicates, ind_to_classes=ind_to_classes)))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes(track_recall_samples=track_recall_samples, ind_to_predicates=ind_to_predicates, ind_to_classes=ind_to_classes)  # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    # 该函数接收一个可迭代对象，返回一个行为与原对象相同的迭代器，但在每次请求值时打印动态更新的进度条。
    prog_bar = tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size),
                    disable=DIS_PROGRESS_BAR, bar_format='{l_bar}{bar}{r_bar}\n')

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
    confusion_matrix_int = evaluator[conf.mode].result_dict['predicate_confusion_matrix_int']
    matrices = None
    if not matrix_eval:
        # matrices; mp(multiple preds) equals `without constraint`
        recall = evaluator[conf.mode].print_stats()
        recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
        mean_recall = calculate_mr(evaluator_list, conf.mode)
        mean_recall_mp = calculate_mr(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)
        matrices = (recall, recall_mp, mean_recall, mean_recall_mp)

        if conf.test:
        #     print('~~~~~~~~ Confusion Matrix in Val Epoch ~~~~~~~~')
        #     print(confusion_matrix_int)
            dump_file = data_path(f'confusion_matrix_{conf.mode}.npy')
            np.save(dump_file, confusion_matrix_int)

            # 保存每个谓词的 R@K 指标到 JSON 文件（仅 PredCls 模式）
            if conf.mode == 'predcls':
                json_file = data_path(f'per_predicate_recall_{conf.mode}.json')
                save_per_predicate_recall_json(
                    evaluator_list, conf.mode, multiple_preds=False,
                    output_file=json_file, k_values=[50, 100]
                )

            # 如果是 PredCls 且启用了召回样本跟踪，保存召回样本的详细信息
            if track_recall_samples and conf.mode == 'predcls':
                recall_samples = evaluator[conf.mode].recall_samples
                json_file_recall = data_path(f'recall_samples_{conf.mode}.json')
                save_recall_samples_json(recall_samples, output_file=json_file_recall)

    model.train()
    return matrices, confusion_matrix
