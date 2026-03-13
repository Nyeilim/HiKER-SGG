"""
轻量级可视化 Demo - 用于快速推理和可视化
"""

import os
import sys
import random

import numpy as np
from pyximport import pyximport

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from visualize.inference import SimpleInference
from visualize.renderer import SceneGraphRenderer

# 导入 DATA_PATH
import config

DATA_PATH = config.DATA_PATH


def run_demo(
        task_type='predcls',
        img_indices=(0, 1, 2),
        ckpt_path=None,
        output_dir=None,
        print_title=True
):
    """
    运行可视化 demo
    """
    # 初始化推理器
    print("\nInitializing inference engine...")
    inference = SimpleInference(task_type=task_type, ckpt_path=ckpt_path)
    # 初始化渲染器
    renderer = SceneGraphRenderer()
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing {len(img_indices)} image(s)...")

    # 可视化每张图片
    for idx in img_indices:
        print(f"\n--- Processing image {idx} ---")
        result = inference(idx)

        img_name = os.path.basename(result['img_path']).replace('.jpg', '')
        title = f"{img_name} (idx={idx})"

        # 保存路径
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"{img_name}_pred.png")

        # 渲染（根据 print_title 决定是否显示标题）
        render_title = title if print_title else None
        renderer.render(result, save_path=save_path, show=True, title=render_title)

        # 打印一些统计信息
        print(f"  Image: {result['img_path']}")
        print(f"  GT objects: {len(result['gt_boxes'])}")
        print(f"  GT relations(not filtered): {len(result['gt_relations'])}")
        print(f"  Pred objects: {len(result['pred_boxes'])}")
        print(f"  Pred relations(filtered): {len(result['pred_relations'])}")


if __name__ == '__main__':
    task_type = 'predcls'
    # 随机选取 10 个数字，范围 0~10000
    img_indices = tuple(random.sample(range(10000), 10))
    ckpt_path = None
    output_dir = os.path.join(DATA_PATH, 'visualization')

    run_demo(task_type, img_indices, ckpt_path, output_dir, print_title=True)
