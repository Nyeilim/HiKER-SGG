
"""
可视化场景图生成中使用的图像扰动效果
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 导入扰动函数
from model.dataloaders.corruptions import (
    gaussian_noise, shot_noise, impulse_noise, speckle_noise,
    gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur,
    fog, frost, snow, spatter,
    contrast, brightness, saturate,
    jpeg_compression, pixelate, elastic_transform,
    sunglare, waterdrop, wildfire_smoke, dust, rain
)

# 定义20种主要的扰动类型和对应的函数
CORRUPTIONS = [
    ('Gaussian Noise', gaussian_noise),
    ('Shot Noise', shot_noise),
    ('Impulse Noise', impulse_noise),
    ('Speckle Noise', speckle_noise),
    ('Gaussian Blur', gaussian_blur),
    ('Glass Blur', glass_blur),
    ('Defocus Blur', defocus_blur),
    ('Motion Blur', motion_blur),
    ('Zoom Blur', zoom_blur),
    ('Fog', fog),
    ('Frost', frost),
    ('Snow', snow),
    ('Spatter', spatter),
    ('Contrast', contrast),
    ('Brightness', brightness),
    ('Saturate', saturate),
    ('JPEG Comp.', jpeg_compression),
    ('Pixelate', pixelate),
    ('Elastic', elastic_transform),
    ('Sun Glare', sunglare),
]

# 额外的扰动类型（如果需要）
EXTRA_CORRUPTIONS = [
    ('Waterdrop', waterdrop),
    ('Wildfire Smoke', wildfire_smoke),
    ('Dust', dust),
    ('Rain', rain),
]

def visualize_corruptions(image_path, output_path=None, severity=1, include_extra=False):
    """
    可视化一张图像的所有扰动效果

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（如果为None，则不保存）
        severity: 扰动强度 (1-5)
        include_extra: 是否包含额外的扰动类型
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')

    # 确定要使用的扰动类型
    corruptions_list = CORRUPTIONS + EXTRA_CORRUPTIONS if include_extra else CORRUPTIONS
    n_corruptions = len(corruptions_list)

    # 创建图形
    ncols = 5
    nrows = (n_corruptions + 1) // ncols + 1  # +1 for original image

    fig = plt.figure(figsize=(ncols * 4, nrows * 3))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    # 显示原始图像
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.set_title('Original', fontsize=12, fontweight='bold')
    ax.axis('off')

    # 对每种扰动应用并显示
    for i, (name, corrupt_fn) in enumerate(corruptions_list):
        row = (i + 1) // ncols
        col = (i + 1) % ncols

        try:
            # 应用扰动
            corrupted = corrupt_fn(img.copy(), severity=severity)
            if isinstance(corrupted, np.ndarray):
                corrupted = Image.fromarray(corrupted.astype(np.uint8))

            # 显示
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(corrupted)
            ax.set_title(name, fontsize=10)
            ax.axis('off')
        except Exception as e:
            print(f"Error applying {name}: {e}")
            # 显示错误信息
            ax = fig.add_subplot(gs[row, col])
            ax.text(0.5, 0.5, f"{name}\nError", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    # 隐藏多余的子图
    total_cells = nrows * ncols
    used_cells = n_corruptions + 1
    for i in range(used_cells, total_cells):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.axis('off')

    plt.tight_layout()

    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    plt.show()


def visualize_single_corruption(image_path, corruption_name, output_path=None, severity=1):
    """
    可视化单个扰动在不同强度下的效果

    Args:
        image_path: 输入图像路径
        corruption_name: 扰动名称（从CORRUPTIONS中选择）
        output_path: 输出图像路径
        severity: 扰动强度 (1-5)
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')

    # 找到对应的扰动函数
    corrupt_fn = None
    for name, fn in CORRUPTIONS + EXTRA_CORRUPTIONS:
        if name.lower() == corruption_name.lower():
            corrupt_fn = fn
            break

    if corrupt_fn is None:
        print(f"Corruption '{corruption_name}' not found. Available options:")
        for name, _ in CORRUPTIONS + EXTRA_CORRUPTIONS:
            print(f"  - {name}")
        return

    # 创建图形 - 显示原始图像和5个强度级别
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 原始图像
    axes[0].imshow(img)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 不同强度
    for sev in range(1, 6):
        try:
            corrupted = corrupt_fn(img.copy(), severity=sev)
            if isinstance(corrupted, np.ndarray):
                corrupted = Image.fromarray(corrupted.astype(np.uint8))
            axes[sev].imshow(corrupted)
            axes[sev].set_title(f'Severity {sev}', fontsize=12)
            axes[sev].axis('off')
        except Exception as e:
            print(f"Error at severity {sev}: {e}")
            axes[sev].text(0.5, 0.5, f"Error", ha='center', va='center',
                          transform=axes[sev].transAxes)
            axes[sev].axis('off')

    # 隐藏最后一个子图（如果有）
    if len(axes) > 6:
        axes[6].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    plt.show()


def list_available_corruptions():
    """列出所有可用的扰动类型"""
    print("Available corruptions:")
    print("\nStandard (20 types):")
    for i, (name, _) in enumerate(CORRUPTIONS, 1):
        print(f"  {i:2d}. {name}")
    print("\nExtra (4 types):")
    for i, (name, _) in enumerate(EXTRA_CORRUPTIONS, 21):
        print(f"  {i:2d}. {name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize image corruptions')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output image path')
    parser.add_argument('--severity', '-s', type=int, default=1,
                       choices=[1, 2, 3, 4, 5],
                       help='Corruption severity (1-5, default: 1)')
    parser.add_argument('--single', type=str, default=None,
                       help='Visualize single corruption with multiple severities')
    parser.add_argument('--extra', action='store_true',
                       help='Include extra corruptions (waterdrop, smoke, dust, rain)')
    parser.add_argument('--list', action='store_true',
                       help='List all available corruptions')

    args = parser.parse_args()

    if args.list:
        list_available_corruptions()
    elif args.single:
        visualize_single_corruption(args.image_path, args.single,
                                args.output, args.severity)
    else:
        visualize_corruptions(args.image_path, args.output,
                         args.severity, args.extra)
