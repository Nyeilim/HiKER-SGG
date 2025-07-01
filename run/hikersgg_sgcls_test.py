import os
import sys

import numpy as np

sys.path.append("/output/HiKER-SGG/")

# 如果把 sg_val/val_fn 放在后面就会导入报错，因为里面有个很重要的 setup 语句能导入 lib.fpn.box_intersections_cpu.bbox
import config
from model.refactor.val_fn import val_epoch
from model.refactor.provider import provide_dataloader, provide_model
from config import ModelConfig, CONF_MAT_UPDATED, CONF_MAT_FREQ_TRAIN, data_path, print_globals
from model.refactor.util import load_best_matrices

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
exp_name = 'hikersgg_sgcls_test'
# 注意：这里需要加载SGCls训练的最佳模型
# 如果best_matrices.json中没有SGCls的结果，可以手动指定epoch
test_epoch = 10  # 手动指定要测试的epoch，或者从best_matrices.json中加载
print(f"Start test epoch {test_epoch}")

# Change ckpt path for the evaluated model
# 注意：这里需要指向SGCls训练生成的模型文件
# If you want to test on VG-C benchmark, include "-test_n" in the command line
conf = ModelConfig(f'''
-m sgcls
-p 2500
-clip 5
-ckpt checkpoints/kern_sgcls/hikersgg_sgcls_train/vgrel-{test_epoch}.tar
-b 3
-nwork 9
-ngpu 1
-lr 1e-5  # 测试时学习率不重要，但需要设置
-pooling_dim 4096
-ggnn_rel_time_step_num 3
-ggnn_rel_hidden_dim 1024
-test
-require_overlap_det
-use_bpl
-use_knowledge
-use_embedding
-filter_duplicate_rels
-refine_obj_cls
''')

# 打印配置
conf.print_self_config()
print_globals(config)

# 设置模型对应的混淆矩阵
# SGCls训练会生成自己的混淆矩阵，需要根据实际训练情况调整
matrix_suffix = test_epoch - (test_epoch + 1) % 3  # 混淆矩阵会每三轮计算一次
if matrix_suffix < 2:
    # 如果还没有生成混淆矩阵，使用初始的混淆矩阵
    conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
else:
    # 使用SGCls训练生成的混淆矩阵
    # 注意：这里可能需要根据实际的混淆矩阵文件路径进行调整
    conf_matrix = np.load(data_path(f'misc/conf/conf_mat_updated_{matrix_suffix}.npy'))
np.save(CONF_MAT_UPDATED, conf_matrix)

test_set, test_set_loader = provide_dataloader(conf, 'test')  # 加载数据集
model = provide_model(conf, test_set.ind_to_classes, test_set.ind_to_predicates)  # 加载模型
val_epoch(model, conf, test_set, test_set_loader)  # 开始评估
# model = finetune(model, conf) # 微调
# val_epoch(model, conf, test_set, test_set_loader) # 再次评估 