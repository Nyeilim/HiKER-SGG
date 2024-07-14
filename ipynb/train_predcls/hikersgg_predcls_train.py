import os

import numpy as np
import torch
from apex import amp

from lib.exp.conf_matrix_fn import train_evaluate
from lib.exp.global_var import detector, optimizer, write, conf
from lib.exp.train_fn import train_epoch
from lib.exp.val_fn import val_epoch
from lib.my_util import adj_normalize

alpha = 0.9
start_epoch = 0
end_epoch = 20
detector, optimizer = amp.initialize(detector, optimizer, opt_level="O0")

conf_matrix_list = []
for epoch in range(start_epoch, end_epoch):
    if (epoch + 1) % 3 == 0:
        print('Evaluating new confusion matrix...')
        conf_matrix = train_evaluate()  # 获取新的谓词混淆矩阵(见 3.7)
        conf_matrix[0, :] = 0.0
        conf_matrix[:, 0] = 0.0
        conf_matrix[0, 0] = 1.0
        conf_matrix = conf_matrix / (conf_matrix.sum(-1)[:, None] + 1e-8)
        conf_matrix = adj_normalize(conf_matrix)
        conf_matrix_list.append(conf_matrix)

        conf_matrix_old = np.load('/output/data/misc/conf_mat_updated.npy')
        conf_matrix_new = conf_matrix_old * alpha + conf_matrix * (1 - alpha)
        np.save('/output/data/misc/conf_mat_updated.npy', conf_matrix_new)
        np.save('/output/data/misc/conf/conf_mat_updated_{}.npy'.format(epoch), conf_matrix_new)

    write(f'epoch = {epoch}')
    # 调整学习率
    if epoch != 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    rez = train_epoch(epoch)  # 开始训练
    losses_mean_epoch = rez.mean(axis=0)
    losses_mean_epoch_class = losses_mean_epoch['loss_class']
    losses_mean_epoch_rel = losses_mean_epoch['loss_rel']
    losses_mean_epoch_total = losses_mean_epoch['loss_total']
    write("overall{:2d}: ({:.3f})\n{}".format(epoch, losses_mean_epoch_total, losses_mean_epoch))

    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            # {k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            'state_dict': detector.state_dict(),
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))
        # noinspection PyPackageRequirements
        print(os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()  # 开始评估
