"""
场景图可视化渲染器
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SceneGraphRenderer:
    """场景图可视化渲染器"""

    def __init__(self, figsize=(16, 12)):
        """
        Args:
            figsize: 图像大小
        """
        self.figsize = figsize

        # 物体框颜色列表（为不同物体分配不同颜色）
        self.box_colors = self._generate_colors(100)

        # 谓词颜色映射（为不同关系类型分配不同颜色）
        self.relation_colors = self._generate_colors(50)

    def _generate_colors(self, n):
        """生成n种不同的颜色"""
        import matplotlib.cm as cm
        colors = cm.tab20(np.linspace(0, 1, min(n, 20)))
        if n > 20:
            colors = np.vstack([colors] * ((n // 20) + 1))
        return colors[:n]

    def render(self, inference_result, save_path=None, show=True, title=None):
        """
        渲染推理结果（单图 Prediction）
        """
        img = inference_result['img']
        pred_boxes = inference_result['pred_boxes']
        pred_classes = inference_result['pred_classes']
        pred_relations = inference_result['pred_relations']
        pred_rel_scores = inference_result['pred_rel_scores']
        bg_probs = inference_result.get('bg_probs', None)
        gt_boxes = inference_result['gt_boxes']
        gt_classes = inference_result['gt_classes']
        gt_relations = inference_result['gt_relations']
        ind_to_classes = inference_result['ind_to_classes']
        ind_to_predicates = inference_result['ind_to_predicates']

        # 1. 获取原图尺寸
        img_width, img_height = img.size

        # 2. 设置 DPI (建议保持 100 或更高，这里保持与你原代码一致)
        dpi = 100

        # 3. 计算 figsize，确保  figsize * dpi = 原图像素
        figsize_width = img_width / dpi
        figsize_height = img_height / dpi

        # 4. 【关键修改】不使用 subplots，直接使用 figure 并添加全覆盖的 axes
        # 这样可以消除 subplots 默认带来的边距，确保 1:1 像素映射
        fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] 均为 0-1 的比例

        # 输出到终端 (保持原有逻辑)
        print(f"\nGround Truth Relations: ")
        for i, (subj_idx, obj_idx, pred_idx) in enumerate(gt_relations):
            if subj_idx == 0 or obj_idx == 0:
                continue
            subj_name = ind_to_classes[gt_classes[subj_idx]]
            obj_name = ind_to_classes[gt_classes[obj_idx]]
            pred_name = ind_to_predicates[pred_idx]
            print(f"   <{subj_name}, {pred_name}, {obj_name}>")

        print(f"\nPrediction Relations: ")
        pred_triplets = pred_relations
        pred_rel_probs_val = pred_rel_scores

        pred_dict = {}
        for i, (subj_idx, obj_idx, pred_idx) in enumerate(pred_triplets):
            if subj_idx != 0 and obj_idx != 0:
                pred_dict[(subj_idx, obj_idx)] = (pred_idx, pred_rel_probs_val[i], bg_probs[i])

        for gt_rel in gt_relations:
            subj_idx, obj_idx, gt_pred_idx = gt_rel
            if subj_idx == 0 or obj_idx == 0:
                continue

            subj_name = ind_to_classes[gt_classes[subj_idx]]
            obj_name = ind_to_classes[gt_classes[obj_idx]]
            gt_pred_name = ind_to_predicates[gt_pred_idx]

            pred_result = pred_dict.get((subj_idx, obj_idx))

            if pred_result is not None:
                pred_idx, pred_prob, bg_prob = pred_result
                pred_name = ind_to_predicates[pred_idx]
                is_correct = (pred_idx == gt_pred_idx)
                status = "✓" if is_correct else "✗"
                print(f"  {status}  <{subj_name}, {pred_name}, {obj_name}> {pred_prob:.2f} (bg: {bg_prob:.2f}) | GT: {gt_pred_name}")
            else:
                print(f"  ✗  <{subj_name}, --, {obj_name}> -- | GT: {gt_pred_name}")

        # 渲染单图
        self._render_single(
            ax, img, pred_boxes, pred_classes, pred_relations,
            ind_to_classes, ind_to_predicates, ''
        )

        if save_path:
            # 5. 【关键修改】保存参数优化
            # pad_inches=0 确保不留白边
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0,
                       facecolor='white', edgecolor='none', optimize=False)
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        plt.close(fig)

    def _render_single(self, ax, img, boxes, classes, relations,
                   ind_to_classes, ind_to_predicates, title, rel_scores=None, bg_probs=None):
        """渲染单个视图"""
        img_width, img_height = img.size

        # 明确设置 extent 和 origin，禁止自动缩放
        # 使用 interpolation='none' 保持原始像素，避免模糊
        ax.imshow(img, extent=[0, img_width, img_height, 0], origin='upper', interpolation='none')
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # 绘制边界框和类别（每个框使用不同颜色）
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # 为每个框分配不同颜色
            box_color = self.box_colors[i % len(self.box_colors)]

            # 绘制矩形框（linewidth=1）
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=1, edgecolor=box_color, facecolor='none'
            )
            ax.add_patch(rect)

            # 绘制类别标签（fontsize=8）
            class_name = ind_to_classes[cls]

            # 【修改1】智能调整标签位置，避免超出图片边界
            # 如果框在图片顶部（y1 < 15），标签放在框内下方；否则放在框上方
            if y1 < 15:
                # 框在顶部，标签放在框内下方
                label_y = y1 + 15
                # 确保不超出图片底部
                if label_y > img_height - 5:
                    label_y = img_height - 5
            else:
                # 框不在顶部，标签放在框上方
                label_y = y1 - 5

            ax.text(x1, label_y, class_name,
                   color=box_color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))