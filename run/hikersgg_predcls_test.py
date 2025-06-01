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
exp_name = 'hikersgg_predcls_test'
test_epoch = load_best_matrices()['best_mr_epoch']  # 需要测试第几个 epoch 训练出来的模型
print(f"Start test epoch {test_epoch}")

# Change ckpt path for the evaluated model
# If you want to test on VG-C benchmark, include "-test_n" in the command line
conf = ModelConfig(f'''
-m predcls
-p 2500
-clip 5
-ckpt checkpoints/kern_predcls/hikersgg_predcls_train/vgrel-{test_epoch}.tar
-b 8
-nwork 8
-ngpu 1
-lr 1e-4
-pooling_dim 4096
-ggnn_rel_time_step_num 3
-ggnn_rel_hidden_dim 1024
-test
-require_overlap_det
-use_bpl
-use_knowledge
-use_embedding
-filter_duplicate_rels
''')

# 打印配置
conf.print_self_config()
print_globals(config)

# 设置模型对应的混淆矩阵
matrix_suffix = test_epoch - (test_epoch + 1) % 3  # 混淆矩阵会每三轮计算一次
if matrix_suffix < 2:
    conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
else:
    conf_matrix = np.load(data_path(f'misc/conf/conf_mat_updated_{matrix_suffix}.npy'))
np.save(CONF_MAT_UPDATED, conf_matrix)

test_set, test_set_loader = provide_dataloader(conf, 'test')  # 加载数据集
model = provide_model(conf, test_set.ind_to_classes, test_set.ind_to_predicates)  # 加载模型
val_epoch(model, conf, test_set, test_set_loader)  # 开始评估
# model = finetune(model, conf) # 微调
# val_epoch(model, conf, test_set, test_set_loader) # 再次评估
