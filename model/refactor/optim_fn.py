from apex.optimizers import FusedAdam, FusedSGD
from config import logger


# 优化器，用于执行梯度下降
def get_optim(model, conf):

    # 学习率的大小应该和 num_gpus、batch_size 数成比例关系
    real_lr = conf.lr * conf.num_gpus * conf.batch_size
    logger.info("real_lr: {}".format(real_lr))

    # Lower the learning rate on the VGG fully connected layers by 1/10th.
    # It's a hack, but it helps stabilize the models.
    fc_params = [p for n, p in model.named_parameters() if
                 (n.startswith('roi_fmap') or 'clean' in n) and p.requires_grad]
    non_fc_params = [p for n, p in model.named_parameters() if
                     not (n.startswith('roi_fmap') or 'clean' in n) and p.requires_grad]
    params = [{'params': fc_params, 'lr': real_lr / 10.0}, {'params': non_fc_params}]

    if conf.adam:
        optimizer = FusedAdam(params, weight_decay=conf.adamwd, lr=real_lr, eps=1e-3)
    else:
        optimizer = FusedSGD(params, weight_decay=conf.l2, lr=real_lr, momentum=0.9)

    return optimizer
