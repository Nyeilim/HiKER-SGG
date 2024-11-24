import numpy as np
import pyximport
import sys

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
# 这行语句会自动编译项目里 .pyx 文件，这里会导致两次编译，分别是 box_intersections_cpu 和 draw_rectangles
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
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

    # 翻译后的，每行的 <s,o,p> 分别对应着 entity.index, predicate.index
    gt_rels_trans = [[gt_classes[item[0], gt_classes[item[1], item[2]]]] for item in gt_rels]
    edge_matrix[gt_rels_trans[0]][gt_rels_trans[1]][gt_rels_trans[2]] += 1

np.save(file, edge_matrix)