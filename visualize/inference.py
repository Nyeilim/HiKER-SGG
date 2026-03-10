"""
轻量级推理模块 - 只进行模型推理，不计算指标或记录时间
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import ModelConfig, BOX_SCALE, IM_SCALE, data_path
from model.refactor.provider import provide_model, provide_dataloader
from model.refactor.util import load_best_matrices


class SimpleInference:
    """轻量级推理类 - 只进行推理，不做评估"""

    def __init__(self, task_type='predcls', ckpt_path=None, test_n=False):
        """
        初始化推理器

        Args:
            task_type: 任务类型 ('predcls' 或 'sgcls')
            ckpt_path: 模型checkpoint路径（None 表示使用最优模型）
            test_n: 是否使用带扰动的测试集
        """
        self.task_type = task_type
        self.test_n = test_n

        # 使用 create_test_config 创建配置，这个方法已经包含了最优 epoch 和混淆矩阵的处理
        self.conf = create_test_config(task_type, epoch=None if ckpt_path is None else None)

        # 如果指定了自定义 ckpt_path，设置到配置
        if ckpt_path is not None:
            self.conf.ckpt = ckpt_path

        # 设置测试模式
        if test_n:
            self.conf.test_n = True

        # 加载数据集和模型
        print("Loading dataset...")
        self.dataset, self.dataloader = provide_dataloader(self.conf, 'test')

        print("Loading model...")
        self.model = provide_model(
            self.conf,
            self.dataset.ind_to_classes,
            self.dataset.ind_to_predicates
        )

        # 获取谓词映射
        self.ind_to_predicates = self.dataset.ind_to_predicates

        print(f"Dataset size: {len(self.dataset)}")
        print(f"Model loaded: {task_type}")
        print(f"Test with corruptions: {test_n}")

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
        for idx, batch in enumerate(self.dataloader):
            if idx == img_idx:
                break

        # 推理
        with torch.no_grad():
            det_res = self.model(batch)

        if self.conf.num_gpus == 1:
            det_res = [det_res]

        # 解析结果
        boxes, objs, obj_scores, rels, pred_scores = det_res[0]

        # 获取ground truth
        img_path = self.dataset.filenames[img_idx]
        img = Image.open(img_path).convert('RGB')

        gt_boxes = self.dataset.gt_boxes[img_idx].copy()
        gt_classes = self.dataset.gt_classes[img_idx].copy()
        gt_relations = self.dataset.relationships[img_idx].copy()

        return {
            'img': img,
            'img_path': img_path,
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'gt_relations': gt_relations,
            'pred_boxes': boxes * BOX_SCALE/IM_SCALE,
            'pred_classes': objs,
            'pred_relations': rels,
            'pred_rel_scores': pred_scores,
            'ind_to_classes': self.dataset.ind_to_classes,
            'ind_to_predicates': self.ind_to_predicates,
        }

    def get_class_name(self, class_idx):
        """获取类别名称"""
        return self.dataset.ind_to_classes[class_idx]

    def get_predicate_name(self, pred_idx):
        """获取谓词名称"""
        return self.ind_to_predicates[pred_idx]
