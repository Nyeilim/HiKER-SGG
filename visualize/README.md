# HiKER-SGG 轻量级可视化工具

一个简化版的可视化工具，用于快速推理和显示场景图生成结果，不包含评估指标计算和性能计时等冗余功能。

## 特性

- **轻量级**: 只进行模型推理，不做评估
- **快速**: 直接调用模型，无额外的计算开销
- **直观**: 并排显示 Ground Truth 和 Prediction
- **灵活**: 支持单图和多图可视化

## 包结构

```
visualize/
├── __init__.py       # 包初始化
├── inference.py       # 简化推理逻辑
├── renderer.py       # 场景图可视化渲染器
└── demo.py          # 命令行入口
```

## 安装依赖

确保你的环境已安装：
```bash
pip install torch torchvision numpy matplotlib Pillow
```

## 使用方法

### 1. 基本用法 - 可视化前3张图片

```bash
python -m visualize.demo --task predcls
```

### 2. 指定图片索引

```bash
python -m visualize.demo --task predcls --indices 0 5 10 20
```

### 3. 使用自定义模型

```bash
python -m visualize.demo --task predcls --ckpt path/to/checkpoint.tar
```

> **注意**: 不指定 `--ckpt` 时，会自动加载 `best_matrices.json` 中记录的最优 epoch 模型。

### 4. 测试带扰动的数据集 (VG-C)

```bash
python -m visualize.demo --task predcls --test_n
```

### 5. 保存可视化结果

```bash
python -m visualize.demo --task predcls --output ./visualizations
```

### 6. 组合选项

```bash
python -m visualize.demo \
    --task sgcls \
    --ckpt checkpoints/kern_sgcls/hikersgg_sgcls_train/vgrel-15.tar \
    --indices 0 1 2 3 4 \
    --output ./results \
    --test_n
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|--------|------|--------|
| `--task` | `-t` | 任务类型 (predcls 或 sgcls) | predcls |
| `--indices` | `-i` | 要可视化的图片索引 (空格分隔) | [0, 1, 2] |
| `--ckpt` | `-c` | 模型 checkpoint 路径（None 表示加载最优模型）| 最优 epoch |
| `--test_n` | - | 使用带扰动的测试集 | False |
| `--output` | `-o` | 输出目录 | None (只显示不保存) |
| `--single` | - | 单图模式 | False |

## Python API 使用

```python
from visualize.inference import SimpleInference
from visualize.renderer import SceneGraphRenderer

# 初始化推理器（自动加载最优模型）
inference = SimpleInference(task_type='predcls')

# 或指定特定模型
# inference = SimpleInference(task_type='predcls', ckpt_path='path/to/checkpoint.tar')

# 初始化渲染器
renderer = SceneGraphRenderer()

# 推理并可视化
result = inference(img_idx=0)
renderer.render(result, save_path='output.png', show=True)
```

## 可视化说明

- **左侧图像**: Ground Truth（真实标注）
- **右侧图像**: Prediction（模型预测）
- **绿色矩形框**: 物体边界框
- **彩色线条**: 物体间的关系
- **标签**:
  - 上方绿色文字：物体类别
  - 线条中央文字：关系类型和置信度

## 支持的任务

1. **PredCls** (Predicate Classification)
   - 物体框为 Ground Truth
   - 只预测关系类型

2. **SGCls** (Scene Graph Classification)
   - 物体框为 Ground Truth
   - 预测物体类别和关系类型

## 注意事项

1. 确保数据集路径在 `config.py` 中正确配置
2. 确保模型 checkpoint 路径正确
3. 如使用 `-test_n`，需要确保 `corruptions.py` 中的图像资源文件存在

## 与原测试代码的区别

| 功能 | 原测试代码 | 轻量级版本 |
|------|------------|-------------|
| 模型推理 | ✓ | ✓ |
| 指标计算 | ✓ | ✗ |
| 性能计时 | ✓ | ✗ |
| 混淆矩阵更新 | ✓ | ✗ |
| 可视化 | ✗ | ✓ |
| 代码复杂度 | 高 | 低 |
