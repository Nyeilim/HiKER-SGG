"""
轻量级可视化 Demo - 用于快速推理和可视化
"""

import os
import sys
import numpy as np
from pyximport import pyximport

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.refactor.provider import provide_model, provide_dataloader
from model.refactor.util import load_best_matrices
from visualize.inference import SimpleInference
from visualize.renderer import SceneGraphRenderer

# 导入 DATA_PATH
import config

DATA_PATH = config.DATA_PATH


def run_demo(
        task_type='predcls',
        img_indices=(0, 1, 2),
        ckpt_path=None,
        test_n=False,
        output_dir=None
):
    """
    运行可视化 demo
    """
    # 初始化推理器
    print("\nInitializing inference engine...")
    inference = SimpleInference(task_type=task_type, ckpt_path=ckpt_path, test_n=test_n)
    # 初始化渲染器
    renderer = SceneGraphRenderer()
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nVisualizing {len(img_indices)} image(s)...")

    # 可视化每张图片
    for idx in img_indices:
        print(f"\n--- Processing image {idx} ---")
        result = inference(idx)

        img_name = os.path.basename(result['img_path']).replace('.jpg', '')
        title = f"{img_name} (idx={idx})"

        # 保存路径
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{img_name}_comparison.png")

        # 渲染
        renderer.render(result, save_path=save_path, show=True, title=title)

        # 打印一些统计信息
        print(f"  Image: {result['img_path']}")
        print(f"  GT objects: {len(result['gt_boxes'])}")
        print(f"  GT relations: {len(result['gt_relations'])}")
        print(f"  Pred objects: {len(result['pred_boxes'])}")
        print(f"  Pred relations: {len(result['pred_relations'])}")


if __name__ == '__main__':
    task_type = 'predcls'
    img_indices = (0, 1, 2)
    ckpt_path = None
    test_n = False
    output_dir = os.path.join(DATA_PATH, 'visualization')

    run_demo(task_type, img_indices, ckpt_path, test_n, output_dir)
