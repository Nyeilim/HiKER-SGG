from collections import defaultdict, OrderedDict

import numpy as np
import pyximport
import sys

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
# 这行语句会自动编译项目里 .pyx 文件，这里会导致两次编译，分别是 box_intersections_cpu 和 draw_rectangles
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from config import ModelConfig, EDGE_MATRIX
from lib.exp.provider import provide_dataloader

# 获取训练集中初始边信息，这是个可以单独运行的程序
# 构建个 151x151x51 的数组，对应 151 个实体间的 51 种关系
edge_matrix = np.zeros((151,151,51), dtype=int)
file = EDGE_MATRIX
non_rel_count, bg_count = 0, 0
verbose = False
conf = ModelConfig(f'''
-val_size 5000
-filter_duplicate_rels
-m predcls
-b 8
-ngpu 1
-nwork 24
''')

train_full, train_full_loader = provide_dataloader(conf, 'confusion_matrix_val')
# test, test_loader = provide_dataloader(conf, 'test')
ind_to_classes, ind_to_predicates = train_full.ind_to_classes, train_full.ind_to_predicates
all_pred_in_train = defaultdict(int)
all_pred_in_test = defaultdict(int)
all_pred_in_dataset = defaultdict(int)
train_img_count, test_img_count, all_img_count = 0,0,0
train_rels_count, test_rels_count, all_rels_count = 0,0,0

for entry in train_full:
    gt_rels = entry['gt_relations']
    gt_classes = entry['gt_classes']
    train_img_count += 1
    all_img_count += 1

    for rel in gt_rels:
        s = gt_classes[rel[0]]
        o = gt_classes[rel[1]]
        p = rel[2]
        edge_matrix[s][o][p] += 1

        _subject = ind_to_classes[s]
        _predicate = ind_to_predicates[p]
        _object = ind_to_classes[o]
        all_pred_in_train[_predicate] += 1
        all_pred_in_dataset[_predicate] += 1
        train_rels_count += 1
        all_rels_count += 1

        if verbose:
            print('<{},{},{}>'.format(_subject, _predicate, _object))

# for entry in test:
#     gt_rels = entry['gt_relations']
#     gt_classes = entry['gt_classes']
#     test_img_count += 1
#     all_img_count += 1
#
#     for rel in gt_rels:
#         s = gt_classes[rel[0]]
#         o = gt_classes[rel[1]]
#         p = rel[2]
#         edge_matrix[s][o][p] += 1
#
#         _subject = ind_to_classes[s]
#         _predicate = ind_to_predicates[p]
#         _object = ind_to_classes[o]
#         all_pred_in_test[_predicate] += 1
#         all_pred_in_dataset[_predicate] += 1
#         test_rels_count += 1
#         all_rels_count += 1
#
#         if verbose:
#             print('<{},{},{}>'.format(_subject, _predicate, _object))

all_pred_in_train = OrderedDict(sorted(all_pred_in_train.items(), key=lambda item: item[1], reverse=True))
# all_pred_in_test = OrderedDict(sorted(all_pred_in_test.items(), key=lambda item: item[1], reverse=True))
# all_pred_in_dataset = OrderedDict(sorted(all_pred_in_dataset.items(), key=lambda item: item[1], reverse=True))

print('train | images:{} rels:{} predicates:{}'.format(train_img_count, train_rels_count, all_pred_in_train))
# print('test | images:{} rels:{} predicates:{}'.format(test_img_count, test_rels_count, all_pred_in_test))
# print('all | images:{} rels:{} predicates:{}'.format(all_img_count, all_rels_count, all_pred_in_dataset))
# np.save(file, edge_matrix)