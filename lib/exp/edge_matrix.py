import numpy as np
import pyximport
import sys

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
# 这行语句会自动编译项目里 .pyx 文件，这里会导致两次编译，分别是 box_intersections_cpu 和 draw_rectangles
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from config import data_path, ModelConfig
from lib.exp.provider import provide_dataloader

# 获取训练集中初始边信息，这是个可以单独运行的程序
# 构建个 151x151x51 的数组，对应 151 个实体间的 51 种关系
edge_matrix = np.zeros((151,151,51))
file = data_path('edge_matrix.npy')
non_rel_count, bg_count = 0, 0
verbose = True
conf = ModelConfig(f'''
-val_size 5000
-filter_duplicate_rels
-m predcls
-b 8
-ngpu 1
-nwork 24
''')

train_full, train_full_loader = provide_dataloader(conf, 'train')
ind_to_classes, ind_to_predicates = train_full.ind_to_classes, train_full.ind_to_predicates

for entry in train_full:
    gt_rels = entry['gt_relations']
    gt_classes = entry['gt_classes']

    for rel in gt_rels:
        s = gt_classes[rel[0]]
        o = gt_classes[rel[1]]
        p = rel[2]
        edge_matrix[s][o][p] += 1

        if verbose:
            print('<{},{},{}>'.format(ind_to_classes[s], ind_to_predicates[p], ind_to_classes[o]))
        if 0 == p:
            non_rel_count += 1
        if 0 == s or 0 == o:
            bg_count += 1

print('non_rel_count:{}, bg_count:{}'.format(non_rel_count, bg_count))
np.save(file, edge_matrix)