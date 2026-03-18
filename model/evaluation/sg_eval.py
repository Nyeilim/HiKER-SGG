"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
from math import isnan as math_isnan
from functools import reduce
from pickle import dump as pickle_dump
from numpy import mean as np_mean, column_stack as np_column_stack, \
                  ones as np_ones, zeros as np_zeros, union1d as np_union1d, \
                  all as np_all, where as np_where, \
                  in1d as np_in1d, concatenate as np_concatenate, \
                  save as np_save, set_printoptions as np_set_printoptions
from model.pytorch_misc import intersect_2d, argsort_desc

import numpy
import pyximport
# numpy.get_include(): '/output/hiker-sgg/lib/python3.8/site-packages/numpy/core/include'
# 这行语句会自动编译项目里 .pyx 文件，这里会导致两次编译，分别是 box_intersections_cpu 和 draw_rectangles
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps    # 这行在首次运行时会在终端弹出编译信息
from config import MODES
np_set_printoptions(precision=3)


class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False, track_recall_samples=False, ind_to_predicates=None, ind_to_classes=None):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
        self.result_dict['predicate_confusion_matrix'] = np_zeros([51, 51], dtype='float')
        self.result_dict['predicate_confusion_matrix_int'] = np_zeros([51, 51], dtype='int')
        self.multiple_preds = multiple_preds

        # 跟踪召回样本的置信度信息
        self.track_recall_samples = track_recall_samples
        self.ind_to_predicates = ind_to_predicates
        self.ind_to_classes = ind_to_classes
        if track_recall_samples:
            self.recall_samples = {}  # {predicate_name: [list of recalled samples]}
            # 初始化每个谓词的列表
            if ind_to_predicates is not None:
                for pred_name in ind_to_predicates:
                    self.recall_samples[pred_name] = []

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_entry, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                                  viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds,
                                  track_recall_samples=self.track_recall_samples,
                                  recall_samples=self.recall_samples if self.track_recall_samples else None,
                                  ind_to_predicates=self.ind_to_predicates,
                                  ind_to_classes=self.ind_to_classes)
        # self.print_stats()
        return res

    def save(self, fn):
        np_save(fn, self.result_dict)

    def print_stats(self):
        if self.multiple_preds:
            recall_method = 'recall without constraint'
        else:
            recall_method = 'recall with constraint'
        output = {}
        print(f'======================{self.mode}  {recall_method}============================', flush=True)
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np_mean(v)), flush=True)
            output['R@%i' % k] = np_mean(v)
        return output

    def get_stats(self):
        recall_method = 'recall without constraint' if self.multiple_preds else 'recall with constraint'
        output = {}
        for k, v in self.result_dict[self.mode + '_recall'].items():
            output['R@%i' % k] = np_mean(v)
        return output


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, multiple_preds=False,
                       viz_dict=None, track_recall_samples=False, recall_samples=None, ind_to_predicates=None, ind_to_classes=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict:
    :param viz_dict:
    :param track_recall_samples: Whether to track recalled samples with confidence
    :param recall_samples: Dictionary to store recalled samples by predicate
    :param ind_to_predicates: List of predicate names indexed by predicate ID
    :param ind_to_classes: List of class names indexed by class ID
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']

    if mode == 'predcls':
        pred_boxes = gt_boxes
        pred_classes = gt_classes
        obj_scores = np_ones(gt_classes.shape[0])
    elif mode == 'sgcls':
        pred_boxes = gt_boxes
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'sgdet' or mode == 'phrdet':
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np_column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    else:
        raise ValueError('invalid mode')

    if multiple_preds:
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np_column_stack((pred_rel_inds[score_inds[:,0]], score_inds[:,1]+1))
        predicate_scores = rel_scores[score_inds[:,0], score_inds[:,1]+1]
    else:
        pred_rels = np_column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        predicate_scores = rel_scores[:,1:].max(1)

    pred_to_gt, pred_5ples, rel_scores2 = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)

    confusion_matrix(gt_rels, gt_boxes, gt_classes, pred_rel_inds, rel_scores, result_dict)

    # print("pred_to_gt:", pred_to_gt[:20])
    # print(reduce(np_union1d, pred_to_gt[:20]))
    for k in result_dict[mode + '_recall']:
        match = reduce(np_union1d, pred_to_gt[:k])
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)

    # 跟踪召回样本的置信度信息
    if track_recall_samples and recall_samples is not None:
        # 获取被召回的 GT 关系索引
        recalled_gt_indices = reduce(np_union1d, pred_to_gt[:100])  # 使用 top 100

        # 为每个被召回的 GT 关系记录详细信息
        for gt_idx in recalled_gt_indices:
            gt_idx_int = int(gt_idx)  # 转换为 Python 整数
            gt_rel = gt_rels[gt_idx_int]  # [subject_idx, object_idx, predicate_idx]
            predicate_idx = int(gt_rel[2])

            # 使用 ind_to_predicates 获取谓词名称
            if ind_to_predicates is not None and predicate_idx < len(ind_to_predicates):
                predicate_name = ind_to_predicates[predicate_idx]
            else:
                continue  # 跳过无效的谓词索引

            # 找到预测这个 GT 关系的预测索引
            pred_idx = None
            for i, gt_matches in enumerate(pred_to_gt):
                if gt_idx_int in gt_matches:
                    pred_idx = i
                    break

            if pred_idx is not None:
                # 获取预测信息
                subject_idx = int(pred_rels[pred_idx][0])
                object_idx = int(pred_rels[pred_idx][1])
                predicate_score = float(predicate_scores[pred_idx])

                # 获取主体和客体的类别名称（将索引转换为名称）
                subject_class_idx = int(pred_classes[subject_idx])
                object_class_idx = int(pred_classes[object_idx])

                if ind_to_classes is not None:
                    subject_class = ind_to_classes[subject_class_idx] if subject_class_idx < len(ind_to_classes) else str(subject_class_idx)
                    object_class = ind_to_classes[object_class_idx] if object_class_idx < len(ind_to_classes) else str(object_class_idx)
                else:
                    subject_class = str(subject_class_idx)
                    object_class = str(object_class_idx)

                # 获取 image_id（如果有的话）
                image_id = gt_entry.get('image_id', 'unknown')

                # 构建召回样本记录
                recall_sample = {
                    "image_id": str(image_id),
                    "subject": subject_class,
                    "object": object_class,
                    "confidence": float(predicate_score)
                }

                # 添加到对应谓词的列表中
                if predicate_name in recall_samples:
                    recall_samples[predicate_name].append(recall_sample)

    return pred_to_gt, pred_5ples, rel_scores2

def confusion_matrix(gt_rels, gt_boxes, gt_classes, pred_rel_inds, rel_scores, result_dict):
    """
    计算混淆矩阵，存储在 result_dict['predicate_confusion_matrix']
    :param gt_rels: 存储着真实的 <s,o,p> 三元组
    :param gt_boxes: 未使用
    :param gt_classes: 未使用
    :param pred_rel_inds: 存储着预测的 <s,o> 对
    :param rel_scores: 存储着每个 <s,o> 对被预测为 51 个谓词的概率
    :param result_dict: 混淆矩阵的存储结果在其中
    """
    pred_pair_idx = pred_rel_inds[:, 0] * 1024 + pred_rel_inds[:, 1] # 利用预测的 s,o 生成唯一标识符，长度记作 num_prediction
    gt_pair_idx = gt_rels[:, 0] * 1024 + gt_rels[:, 1] # 利用真实的 s,o 生成唯一标识符，生成规则和上面相同，长度记作 num_gt_rel
    # 生成形状 (num_prediction,num_gt_rel) 预测矩阵，其上的 True 表示预测的 <s,o> 同真实的 <s,o> 相匹配
    # np_where 作用在上面，将会返回两个数组，分别代表 True 的地方的行索引和列索引，这个索引会和 pred_rels gt_rels 的索引匹配上
    pred_pair_in_gt = np_where(pred_pair_idx[:, None] == gt_pair_idx[None, :])

    # 剔除空关系，找到最大的谓词概率作为预测谓词，然后形成 <s,o,p> 预测三元组
    pred_rels = np_column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
    pred_scores = rel_scores[:, 1:].max(1)
    pred_inds = pred_pair_in_gt[0] # 预测矩阵 True 项的行索引
    gt_inds = pred_pair_in_gt[1] # 预测矩阵 True 项的列索引
    # match the subject and object

    # if self.mode == 'predcls':.
    for i in range(len(pred_inds)):
        pred_ind = pred_inds[i] # x
        gt_ind = gt_inds[i] # y
        pred_pred_i = pred_rels[pred_ind][2] # 预测谓词
        gt_pred_i = gt_rels[gt_ind][2] # 实际谓词

        # 索引越界检查，有必要吗？
        if (pred_pred_i < result_dict['predicate_confusion_matrix'].shape[1]
                and gt_pred_i < result_dict['predicate_confusion_matrix'].shape[0]):
            result_dict['predicate_confusion_matrix'][gt_pred_i][pred_pred_i] += 1 # 混淆矩阵计数 +1
            result_dict['predicate_confusion_matrix_int'][gt_pred_i][pred_pred_i] += 1

def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np_zeros((0,5)), np_zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    # print(pred_rels[:,:2].max())
    # print(pred_classes.shape[0])
    # print(pred_rels[:,:2], pred_classes)
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np_all(pred_rels[:,0] != pred_rels[:,1])
    assert np_all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    scores_overall = relation_scores.prod(1)
    if not np_all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np_column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np_column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np_column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np_column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np_where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np_concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np_concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np_where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def calculate_mr(evaluator_list, mode, multiple_preds=False, save_file=None, return_per_class=False, predicate_names=None):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        #print('\n')
        #print('relationship: ', pred_name)
        all_rel_results[pred_name] = evaluator_rel[mode].get_stats()

    mean_recall = {}
    mR20 = mR50 = mR100 = 0.0
    for key, value in all_rel_results.items():
        if math_isnan(value['R@100']):
            continue
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']

    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall

    recall_mode = 'mean recall without constraint' if multiple_preds else 'mean recall with constraint'

    print(f'======================{mode}  {recall_mode}============================', flush=True)
    print('mR@20: ', mR20, flush=True)
    print('mR@50: ', mR50, flush=True)
    print('mR@100: ', mR100, flush=True)

    if save_file is not None:
        if multiple_preds:
            save_file = save_file.replace('.pkl', '_multiple_preds.pkl')
        with open(save_file, 'wb') as f:
            pickle_dump(all_rel_results, f)

    if return_per_class is True:
        per_class_recall = {key: [
            all_rel_results[pred_name][key] for (pred_id, pred_name, evaluator_rel) in evaluator_list
        ] for key in ['R@20', 'R@50', 'R@100']}
        return mean_recall, per_class_recall
    return mean_recall


def save_per_predicate_recall_json(evaluator_list, mode, multiple_preds=False, output_file='per_predicate_recall.json', k_values=[50, 100]):
    """
    保存每个谓词的 R@K 指标到 JSON 文件
    :param evaluator_list: 评估器列表 [(pred_id, pred_name, evaluator_rel), ...]
    :param mode: 评估模式 ('predcls', 'sgcls', 'sgdet')
    :param multiple_preds: 是否为多预测模式
    :param output_file: 输出 JSON 文件路径
    :param k_values: 要保存的 R@K 值列表
    :return: 保存的字典
    """
    import json
    from numpy import std as np_std, isnan as np_isnan

    all_rel_results = {}
    predicate_list = []

    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        stats = evaluator_rel[mode].get_stats()
        all_rel_results[pred_name] = stats

        # 构建 per_predicate 列表
        pred_data = {"predicate": pred_name}
        for k in k_values:
            recall_key = f'R@{k}'
            if recall_key in stats:
                pred_data[f'recall_{k}'] = stats[recall_key]
            else:
                pred_data[f'recall_{k}'] = 0.0
        predicate_list.append(pred_data)

    # 计算平均 recall
    mean_recall = {}
    std_recall = {}
    for k in k_values:
        recall_key = f'R@{k}'
        recall_values = []
        for pred_name, stats in all_rel_results.items():
            if recall_key in stats and not np_isnan(stats[recall_key]):
                recall_values.append(stats[recall_key])

        if len(recall_values) > 0:
            mean_recall[recall_key] = np_mean(recall_values)
            std_recall[recall_key] = np_std(recall_values)
        else:
            mean_recall[recall_key] = 0.0
            std_recall[recall_key] = 0.0

    # 添加 overall_stats
    overall_stats = {}
    for k in k_values:
        overall_stats[f'mean_recall_{k}'] = float(mean_recall[f'R@{k}'])
        overall_stats[f'std_recall_{k}'] = float(std_recall[f'R@{k}'])

    # 构建 JSON 输出
    output_dict = {
        "overall_stats": overall_stats,
        "per_predicate": predicate_list
    }

    # 保存到 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f'Per-predicate recall saved to {output_file}', flush=True)

    return output_dict


def save_recall_samples_json(recall_samples_dict, output_file='recall_samples.json'):
    """
    保存召回样本的详细信息到 JSON 文件
    :param recall_samples_dict: 召回样本字典 {predicate_name: [list of samples]}
    :param output_file: 输出 JSON 文件路径
    """
    import json
    from numpy import mean as np_mean, min as np_min, max as np_max

    # 计算每个谓词的统计信息
    stats = {}
    for predicate_name, samples in recall_samples_dict.items():
        if len(samples) == 0:
            stats[predicate_name] = {
                "count": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }
        else:
            confidences = [s["confidence"] for s in samples]
            stats[predicate_name] = {
                "count": len(samples),
                "avg_confidence": float(np_mean(confidences)),
                "min_confidence": float(np_min(confidences)),
                "max_confidence": float(np_max(confidences))
            }

    # 构建 JSON 输出
    output_dict = {
        "by_predicate": recall_samples_dict,
        "stats": stats
    }

    # 保存到 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f'Recall samples saved to {output_file}', flush=True)

    return output_dict


def eval_entry(mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    evaluator[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    evaluator_multiple_preds[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    for (pred_id, _, evaluator_rel), (_, _, evaluator_rel_mp) in zip(evaluator_list, evaluator_multiple_preds_list):
        gt_entry_rel = gt_entry.copy()
        mask = np_in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
        if gt_entry_rel['gt_relations'].shape[0] == 0:
            continue

        evaluator_rel[mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
        )
        evaluator_rel_mp[mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
        )
