import numpy as np

from config import data_path, ModelConfig
from lib.exp.provider import provide_dataloader

# 获取训练集中初始边信息，这是个可以单独运行的程序
# 构建个 51x51x151 的数组，对应 51 个实体间的 151 种关系
edge_matrix = np.zeros((51,51,151))
file = data_path('edge_matrix.npy')
conf = ModelConfig(f'''
-val_size 5000
-filter_duplicate_rels
-m predcls
-b 8
-ngpu 1
-nwork 24
''')

train_full, train_full_loader = provide_dataloader(conf, 'train')
for entry in train_full:
    gt_rels = entry['gt_relations']
    gt_boxes = entry['gt_boxes']
    gt_classes = entry['gt_classes']