import os
import sys

import numpy as np
import torch
from apex import amp
from tqdm import tqdm

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量

import config
from model.refactor.val_fn import val_epoch, confusion_matrix_evaluate
from model.refactor.provider import provide_model, provide_dataloader
from model.refactor.util import save_best_matrices
from model.refactor.optim_fn import get_optim
from model.refactor.train_fn import train_epoch
from model.util import adj_normalize
from config import data_path, CONF_MAT_UPDATED, CONF_MAT_FREQ_TRAIN, ModelConfig, ALPHA, print_globals

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择显卡
exp_name = 'hikersgg_predcls_train'  # 实验名
write = tqdm.write  # 函数引用赋值，用来打印日志

# 创建配置类，加载配置
# vgrel-11 是 GB-Net 提供的预训练模型，HiKER-SGG 的模型结构和 GB-Net 非常接近
conf = ModelConfig(f'''
-m predcls
-p 2500
-clip 5
-tb_log_dir summaries/kern_predcls/{exp_name}
-save_dir checkpoints/kern_predcls/{exp_name}
-ckpt checkpoints/vgdet/vgrel-11.tar
-val_size 5000
-b 8
-nwork 8
-ngpu 1
-lr 1e-4
-nepoch 15
-pooling_dim 4096
-ggnn_rel_time_step_num 3
-ggnn_rel_hidden_dim 1024
-adam
-require_overlap_det
-use_bpl
-use_knowledge
-use_embedding
-filter_duplicate_rels
''')

# 打印配置
conf.print_self_config()
print_globals(config)

# Initialize the confusion matrix 初始化混淆矩阵
initial_conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
initial_conf_matrix[0, :] = 0.0
initial_conf_matrix[:, 0] = 0.0
initial_conf_matrix[0, 0] = 1.0
initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
initial_conf_matrix = adj_normalize(initial_conf_matrix)
np.save(CONF_MAT_UPDATED, initial_conf_matrix)  # 这个玩意是拿来算概率转移矩阵的

# 各种变量的创建
train_set, train_set_loader = provide_dataloader(conf, 'train')
val_set, val_set_loader = provide_dataloader(conf, 'val')
matrix_val_set, matrix_val_set_loader = provide_dataloader(conf, 'confusion_matrix_val')
model = provide_model(conf, train_set.ind_to_classes, train_set.ind_to_predicates)
optimizer = get_optim(model, conf)
model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
conf_matrix_list = []
matrices_list = []  # mean recall of each epoch
nc_matrices_list = []  # mean recall without constraint of each epoch

# 正式开始
for epoch in range(conf.num_epochs):
    if (epoch + 1) % 3 == 0:  # 每三轮重新计算一次混淆矩阵，后面的数字为 2,5,8,11
        print('Evaluating new confusion matrix...')
        # 获取新的谓词混淆矩阵(见 3.7)，使用当前模型过一遍完整训练集 train_full
        conf_matrix = confusion_matrix_evaluate(model, conf, matrix_val_set, matrix_val_set_loader)
        conf_matrix[0, :] = 0.0
        conf_matrix[:, 0] = 0.0
        conf_matrix[0, 0] = 1.0
        conf_matrix = conf_matrix / (conf_matrix.sum(-1)[:, None] + 1e-8)
        conf_matrix = adj_normalize(conf_matrix)  # 行归一化，对应公式 (18)
        conf_matrix_list.append(conf_matrix)

        conf_matrix_old = np.load(CONF_MAT_UPDATED)  # 加载上轮 epoch 的转移概率矩阵
        conf_matrix_new = conf_matrix_old * ALPHA + conf_matrix * (1 - ALPHA)  # 对应公式 (20)
        np.save(CONF_MAT_UPDATED, conf_matrix_new)
        np.save(data_path(f'misc/conf/conf_mat_updated_{epoch}.npy'), conf_matrix_new)

    write(f'epoch = {epoch}')
    # 调整学习率
    if epoch != 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    rez = train_epoch(model, conf, train_set, train_set_loader, epoch, optimizer)  # 开始训练
    losses_mean_epoch = rez.mean(axis=0)
    losses_mean_epoch_class = losses_mean_epoch['loss_class']
    losses_mean_epoch_rel = losses_mean_epoch['loss_rel']
    losses_mean_epoch_total = losses_mean_epoch['loss_total']
    write("overall{:2d}: ({:.3f})\n{}".format(epoch, losses_mean_epoch_total, losses_mean_epoch))

    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            # {k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            'state_dict': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))
        # noinspection PyPackageRequirements
        print(os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    matrices = val_epoch(model, conf, val_set, val_set_loader)  # 开始评估
    matrices_list.append(matrices[2]) # mean_recall
    nc_matrices_list.append(matrices[3]) # mean_recall_mp

save_best_matrices(matrices_list, nc_matrices_list)
