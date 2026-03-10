"""
轻量级可视化 Demo - 用于快速推理和可视化
"""

import argparse
import os
import sys
from visualize.inference import SimpleInference
from visualize.renderer import SceneGraphRenderer


def run_demo(task_type='predcls', img_indices=None, ckpt_path=None,
            test_n=False, output_dir=None, single_image=False):
    """
    运行可视化 demo

    Args:
        task_type: 任务类型 ('predcls' 或 'sgcls')
        img_indices: 要可视化的图片索引列表，如 [0, 1, 2]
        ckpt_path: 模型checkpoint路径
        test_n: 是否使用带扰动的测试集
        output_dir: 输出目录
        single_image: 是否为单张图像模式
    """
    print("=" * 60)
    print("HiKER-SGG Lightweight Visualization Demo")
    print("=" * 60)
    print(f"Task type: {task_type}")
    print(f"Checkpoint: {ckpt_path or 'default (epoch 10)'}")
    print(f"Test with corruptions: {test_n}")
    print("=" * 60)

    # 初始化推理器
    print("\nInitializing inference engine...")
    inference = SimpleInference(task_type=task_type, ckpt_path=ckpt_path, test_n=test_n)

    # 初始化渲染器
    renderer = SceneGraphRenderer()

    # 默认可视化前3张图片
    if img_indices is None:
        img_indices = [0, 1, 2]

    # 创建输出目录
    if output_dir:
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

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight visualization demo for HiKER-SGG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first 3 images with default model
  python -m visualize.demo --task predcls

  # Visualize specific images
  python -m visualize.demo --task predcls --indices 0 5 10

  # Use custom checkpoint
  python -m visualize.demo --task predcls --ckpt path/to/checkpoint.tar

  # Test with corruptions
  python -m visualize.demo --task predcls --test_n

  # Save results to directory
  python -m visualize.demo --task predcls --output ./visualizations
        """
    )

    parser.add_argument('--task', '-t', type=str, default='predcls',
                       choices=['predcls', 'sgcls'],
                       help='Task type (default: predcls)')

    parser.add_argument('--indices', '-i', type=int, nargs='+', default=None,
                       help='Image indices to visualize (space-separated, e.g., 0 1 2)')

    parser.add_argument('--ckpt', '-c', type=str, default=None,
                       help='Path to model checkpoint')

    parser.add_argument('--test_n', action='store_true',
                       help='Test with corruptions (VG-C benchmark)')

    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for saving visualizations')

    parser.add_argument('--single', action='store_true',
                       help='Single image mode (show one image at a time)')

    args = parser.parse_args()

    run_demo(
        task_type=args.task,
        img_indices=args.indices,
        ckpt_path=args.ckpt,
        test_n=args.test_n,
        output_dir=args.output,
        single_image=args.single
    )


if __name__ == '__main__':
    main()
