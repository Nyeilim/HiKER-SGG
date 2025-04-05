from time import time as time_time

import pandas as pd
from apex import amp
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import DIS_PROGRESS_BAR
from model.pytorch_misc import clip_grad_norm


def train_epoch(model, conf, train_set, train_set_loader, epoch_num, optimizer):
    model.train()
    tr = []
    start = time_time()
    # disable = not verbose，关闭进度条，不然日志文件会很长
    prog_bar = tqdm(enumerate(train_set_loader), total=int(len(train_set) / train_set_loader.batch_size), disable = DIS_PROGRESS_BAR)
    for batch_idx, batch in prog_bar:
        # print(train_batch(batch, verbose=batch_idx % (conf.print_interval*10) == 0))
        result, loss_dict = train_batch(model, conf, batch, optimizer, verbose=batch_idx % (conf.print_interval * 10) == 0)
        tr.append(loss_dict)

        '''
        if batch_idx % 100 == 0:
            print(loss_pd)
            gt = result.rel_labels[:,3].data.cpu().numpy()
            out = result.rel_dists.data.cpu().numpy()
            ind = np.where(gt)[0]
            print(gt[ind])
            print(np.argmax(out[ind], 1))
            print(np.argmax(out[ind, 1:], 1) + 1)
        '''

        if batch_idx % conf.print_interval == 0 and batch_idx >= conf.print_interval:
            mn = pd.DataFrame(tr[-conf.print_interval:]).mean(axis=0)
            time_per_batch = (time_time() - start) / conf.print_interval
            tqdm.write("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, batch_idx, len(train_set_loader), time_per_batch, len(train_set_loader) * time_per_batch / 60))
            tqdm.write(mn.to_string())
            tqdm.write('-----------')
            start = time_time()
    return pd.DataFrame(tr)


def train_batch(model, conf, batch, optimizer, verbose=False):
    optimizer.zero_grad()
    with autocast():
        result = model[batch]
        loss_class = model.obj_loss(result) # refine_obj_cls 为 False 情况下默认返回 0
        loss_rel = model.rel_loss(result)
        loss_scpred = model.scpred_loss(result)
        loss_fcg = model.fcg_loss(result)

        loss = loss_class + loss_rel + loss_scpred + loss_fcg # 成本函数
    with amp.scale_loss(loss, optimizer) as scaled_loss: # 损失缩放，混合精度
        scaled_loss.backward() # 启用反向传播，计算出各个参数的梯度
    clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.grad is not None],  # 所有叶子节点，即 W、B
                   max_norm=conf.clip, verbose=verbose, clip=True) # 梯度裁剪，所有参数梯度 L2 范数的和不能超过 conf.clip
    optimizer.step() # 梯度下降，更新参数
    return result, {
        'loss_class': float(loss_class),
        'loss_rel': float(loss_rel),
        'loss_scpred': float(loss_scpred),
        'loss_fcg': float(loss_fcg),
        'loss_total': float(loss),
    }
