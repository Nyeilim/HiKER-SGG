from apex.optimizers import FusedAdam, FusedSGD

from lib.exp.global_var import detector, conf


# 优化器
def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n, p in detector.named_parameters() if
                 (n.startswith('roi_fmap') or 'clean' in n) and p.requires_grad]
    non_fc_params = [p for n, p in detector.named_parameters() if
                     not (n.startswith('roi_fmap') or 'clean' in n) and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]

    if conf.adam:
        optimizer = FusedAdam(params, weight_decay=conf.adamwd, lr=lr, eps=1e-3)
    else:
        optimizer = FusedSGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    return optimizer  # , scheduler
