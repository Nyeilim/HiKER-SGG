"""
轻量级推理模块 - 只进行模型推理，不计算指标或记录时间
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BOX_SCALE, IM_SCALE
from model.refactor.provider import provide_model, provide_dataloader
from run.util import create_test_config


class SimpleInference:
    """轻量级推理类 - 只进行推理，不做评估"""

    def __init__(self, task_type='predcls', ckpt_path=None):
        """
        初始化推理器

        Args:
            task_type: 任务类型 ('predcls' 或 'sgcls')
            ckpt_path: 模型checkpoint路径（None 表示使用最优模型）
        """
        self.task_type = task_type

        # 使用 create_test_config 创建配置，这个方法已经包含了最优 epoch 和混淆矩阵的处理
        self.conf = create_test_config(task_type, epoch=None if ckpt_path is None else None)

        # 如果指定了自定义 ckpt_path，设置到配置
        if ckpt_path is not None:
            self.conf.ckpt = ckpt_path

        # 设置 num_workers 为 0，避免 Windows 下的多进程问题
        self.conf.num_workers = 0

        # 加载数据集和模型
        print("Loading dataset...")
        self.dataset, self.dataloader = provide_dataloader(self.conf, 'test')

        print("Loading model...")
        self.model = provide_model(
            self.conf,
            self.dataset.ind_to_classes,
            self.dataset.ind_to_predicates
        )
        self.model.eval()
        print("Model.Training:{}".format(self.model.training))

        # 获取谓词映射
        self.ind_to_predicates = self.dataset.ind_to_predicates

        print(f"Dataset size: {len(self.dataset)}")
        print(f"Model loaded: {task_type}")

    def __call__(self, img_idx):
        """
        对指定索引的图片进行推理

        Args:
            img_idx: 数据集中的图片索引

        Returns:
            img: 原始图片 (PIL Image)
            gt_boxes: Ground truth 边界框 (N, 4)
            gt_classes: Ground truth 类别 (N,)
            gt_relations: Ground truth 关系 (M, 3) - (obj_idx, obj_idx, rel_idx)
            pred_boxes: 预测边界框 (N, 4)
            pred_classes: 预测类别 (N,)
            pred_relations: 预测关系 (M, 3)
            pred_rel_scores: 关系得分 (M,)
        """
        # 跳到指定索引
        batch = None
        for idx, batch in enumerate(self.dataloader):
            if idx == img_idx:
                break

        # 推理
        with torch.no_grad():
            boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = self.model[batch]

        # 获取ground truth
        img_path = self.dataset.filenames[img_idx]
        img = Image.open(img_path).convert('RGB')

        img_width, img_height = img.size

        gt_boxes = self.dataset.gt_boxes[img_idx].copy()
        gt_classes = self.dataset.gt_classes[img_idx].copy()
        gt_relations = self.dataset.relationships[img_idx].copy()

        # 对 GT 关系进行去重（去除重复的三元组）
        if len(gt_relations) > 0:
            gt_relations = np.unique(gt_relations, axis=0)

        # 将 GT 框从 BOX_SCALE (1024) 缩放到图片实际尺寸
        # 假设 GT 框是在 1024x1024 尺度上标注的
        gt_boxes_scaled = gt_boxes / BOX_SCALE * max(img_width, img_height)

        # 预测框已经在 IM_SCALE (592) 尺度上，需要缩放到图片实际尺寸
        # 假设 boxes_i 是在 IM_SCALE 上
        pred_boxes_scaled = boxes_i / IM_SCALE * max(img_width, img_height)

        # 根据 GT 的 <s,o> 对过滤预测关系
        pred_relations, pred_rel_scores, bg_probs = self.filter_relation(gt_relations, rels_i, pred_scores_i)

        return {
            'img': img,
            'img_path': img_path,
            'gt_boxes': gt_boxes_scaled,
            'gt_classes': gt_classes,
            'gt_relations': gt_relations,
            'pred_boxes': pred_boxes_scaled,
            'pred_classes': objs_i,
            'pred_relations': pred_relations,
            'pred_rel_scores': pred_rel_scores,
            'bg_probs': bg_probs,
            'ind_to_classes': self.dataset.ind_to_classes,
            'ind_to_predicates': self.ind_to_predicates,
        }

    def filter_relation(self, gt_relations, rels_i, pred_scores_i):
        """
        根据 GT 的 <s,o> 对过滤预测关系，只保留与 GT 匹配的物体对

        Args:
            gt_relations: GT 关系 (N, 3) - [(subj_idx, obj_idx, pred_idx), ...]
            rels_i: 预测关系对索引 (M, 2) - [(subj_idx, obj_idx), ...]
            pred_scores_i: 预测关系分数 (M, num_predicates)

        Returns:
            filtered_rels: 过滤后的关系三元组 (K, 3) - [(subj_idx, obj_idx, pred_idx), ...]
            filtered_scores: 过滤后的关系分数 (K,) - 每个关系的最大谓词概率
            bg_probs: 背景类概率 (K,) - 每个关系的背景类概率
        """
        if len(gt_relations) == 0:
            # 没有GT关系时，取最大谓词索引
            pred_indices = pred_scores_i[:, 1:].argmax(1) + 1
            filtered_rels = np.column_stack((rels_i, pred_indices))
            filtered_scores = pred_scores_i[:, 1:].max(1)
            bg_probs = pred_scores_i[:, 0]
            return filtered_rels, filtered_scores, bg_probs

        # 生成 GT 和预测的物体对标识符
        gt_pair_idx = gt_relations[:, 0] * 1024 + gt_relations[:, 1]
        pred_pair_idx = rels_i[:, 0] * 1024 + rels_i[:, 1]

        # 找到预测中与 GT 匹配的物体对
        pred_in_gt_mask = np.isin(pred_pair_idx, gt_pair_idx)

        # 过滤关系
        filtered_rel_pairs = rels_i[pred_in_gt_mask]
        filtered_pred_scores = pred_scores_i[pred_in_gt_mask]

        # 取最大概率的谓词索引和分数（跳过背景类索引0）
        pred_indices = filtered_pred_scores[:, 1:].argmax(1) + 1
        filtered_scores = filtered_pred_scores[:, 1:].max(1)
        bg_probs = filtered_pred_scores[:, 0]

        # 拼接成三元组
        filtered_rels = np.column_stack((filtered_rel_pairs, pred_indices))

        return filtered_rels, filtered_scores, bg_probs

    def get_class_name(self, class_idx):
        """获取类别名称"""
        return self.dataset.ind_to_classes[class_idx]

    def get_predicate_name(self, pred_idx):
        """获取谓词名称"""
        return self.ind_to_predicates[pred_idx]
