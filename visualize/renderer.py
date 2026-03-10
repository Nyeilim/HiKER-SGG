"""
场景图可视化渲染器
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


class SceneGraphRenderer:
    """场景图可视化渲染器"""

    def __init__(self, figsize=(16, 12)):
        """
        Args:
            figsize: 图像大小
        """
        self.figsize = figsize

        # 谓词颜色映射（为不同关系类型分配不同颜色）
        self.relation_colors = self._generate_colors(50)

    def _generate_colors(self, n):
        """生成n种不同的颜色"""
        import matplotlib.cm as cm
        colors = cm.tab20(np.linspace(0, 1, min(n, 20)))
        if n > 20:
            # 如果超过20种，循环使用
            colors = np.vstack([colors] * ((n // 20) + 1))
        return colors[:n]

    def render(self, inference_result, save_path=None, show=True, title=None):
        """
        渲染推理结果

        Args:
            inference_result: SimpleInference 的返回结果
            save_path: 保存路径
            show: 是否显示
            title: 图像标题
        """
        img = inference_result['img']
        gt_boxes = inference_result['gt_boxes']
        gt_classes = inference_result['gt_classes']
        gt_relations = inference_result['gt_relations']
        pred_boxes = inference_result['pred_boxes']
        pred_classes = inference_result['pred_classes']
        pred_relations = inference_result['pred_relations']
        pred_rel_scores = inference_result['pred_rel_scores']
        ind_to_classes = inference_result['ind_to_classes']
        ind_to_predicates = inference_result['ind_to_predicates']

        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 2, self.figsize[1]))

        # 左图：Ground Truth
        self._render_single(
            axes[0], img, gt_boxes, gt_classes, gt_relations,
            ind_to_classes, ind_to_predicates, 'Ground Truth'
        )

        # 右图：Prediction
        self._render_single(
            axes[1], img, pred_boxes, pred_classes, pred_relations,
            ind_to_classes, ind_to_predicates, 'Prediction',
            rel_scores=pred_rel_scores if pred_relations is not None else None
        )

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    def _render_single(self, ax, img, boxes, classes, relations,
                   ind_to_classes, ind_to_predicates, title, rel_scores=None):
        """渲染单个视图"""
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        img_width, img_height = img.size

        # 绘制边界框和类别
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # 绘制矩形框
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

            # 绘制类别标签
            class_name = ind_to_classes[cls]
            ax.text(x1, y1 - 5, class_name,
                   color='lime', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # 绘制关系
        if relations is not None and len(relations) > 0:
            # relations 的 shape 是 (num_rels, 2)，只包含 (subj_idx, obj_idx)
            # 需要从 pred_rel_scores 中计算最优谓词索引
            if rel_scores is not None and len(relations) == rel_scores.shape[0]:
                # 预测关系：从 rel_scores 中获取最优谓词索引（跳过背景类别索引0）
                pred_indices = rel_scores[:, 1:].argmax(axis=1) + 1
                pred_rel_probs = rel_scores[:, 1:].max(axis=1)  # 谓词概率
                rel_triplets = [(relations[i][0], relations[i][1], pred_indices[i])
                               for i in range(len(relations))]
            else:
                # Ground Truth 关系：relations 本身就是 (num_rels, 3)，包含谓词索引
                rel_triplets = relations
                pred_rel_probs = None

            for i, (subj_idx, obj_idx, pred_idx) in enumerate(rel_triplets):
                # 过滤掉背景
                if subj_idx == 0 or obj_idx == 0:
                    continue

                # 获取边界框中心
                subj_box = boxes[subj_idx]
                obj_box = boxes[obj_idx]

                subj_center = ((subj_box[0] + subj_box[2]) / 2, (subj_box[1] + subj_box[3]) / 2)
                obj_center = ((obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2)

                # 获取关系名称和得分
                pred_name = ind_to_predicates[pred_idx]
                score_text = f" ({pred_rel_probs[i]:.2f})" if pred_rel_probs is not None else ""

                # 绘制连接线
                color = self.relation_colors[pred_idx % len(self.relation_colors)]
                line = Line2D([subj_center[0], obj_center[0]],
                              [subj_center[1], obj_center[1]],
                              linewidth=1.5, color=color, alpha=0.8)
                ax.add_line(line)

                # 绘制关系标签（在线的中点）
                mid_point = ((subj_center[0] + obj_center[0]) / 2,
                           (subj_center[1] + obj_center[1]) / 2)

                # 确保标签在图像范围内
                mid_point = (
                    np.clip(mid_point[0], 10, img_width - 10),
                    np.clip(mid_point[1], 10, img_height - 10)
                )

                ax.text(mid_point[0], mid_point[1], f"{pred_name}{score_text}",
                       color=color, fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    def render_comparison(self, results_list, ncols=4, save_path=None, show=True):
        """
        渲染多个结果的对比

        Args:
            results_list: 推理结果列表
            ncols: 每行显示的图像数量
            save_path: 保存路径
            show: 是否显示
        """
        n_images = len(results_list)
        nrows = (n_images + 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
        axes = axes.flatten() if nrows > 1 else [axes] if isinstance(axes, plt.Axes) else axes

        for idx, result in enumerate(results_list):
            img = result['img']
            pred_boxes = result['pred_boxes']
            pred_classes = result['pred_classes']
            pred_relations = result['pred_relations']
            pred_rel_scores = result['pred_rel_scores']
            ind_to_classes = result['ind_to_classes']
            ind_to_predicates = result['ind_to_predicates']

            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(result.get('title', f'Image {idx}'), fontsize=12)

            # 绘制预测结果
            self._render_single(
                axes[idx], img, pred_boxes, pred_classes, pred_relations,
                ind_to_classes, ind_to_predicates, '',
                rel_scores=pred_rel_scores
            )

        # 隐藏多余的子图
        for idx in range(len(results_list), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        plt.close(fig)
