from time import time as time_time

import pandas as pd
from apex import amp
from torch.cuda.amp import autocast
from tqdm import tqdm

from lib.exp.global_var import conf, model, train, train_loader, write
from lib.pytorch_misc import clip_grad_norm


def train_epoch(epoch_num, optimizer, verbose=False):
    model.train()
    tr = []
    start = time_time()
    # disable = not verbose，关闭进度条，不然日志文件会很长
    prog_bar = tqdm(enumerate(train_loader), total=int(len(train) / train_loader.batch_size), disable = not verbose)
    for b, batch in prog_bar:
        # print(train_batch(batch, verbose=b % (conf.print_interval*10) == 0))
        result, loss_dict = train_batch(batch, optimizer, verbose=b % (conf.print_interval * 10) == 0)
        tr.append(loss_dict)
        '''
        if b % 100 == 0:
            print(loss_pd)
            gt = result.rel_labels[:,3].data.cpu().numpy()
            out = result.rel_dists.data.cpu().numpy()
            ind = np.where(gt)[0]
            print(gt[ind])
            print(np.argmax(out[ind], 1))
            print(np.argmax(out[ind, 1:], 1) + 1)
        '''

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            #             mn = pd.DataFrame([pd.Series(dicty) for dicty in tr[-conf.print_interval:]]).mean(1)
            mn = pd.DataFrame(tr[-conf.print_interval:]).mean(axis=0)
            time_per_batch = (time_time() - start) / conf.print_interval
            write("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            write(mn.to_string())
            write('-----------')
            start = time_time()
    return pd.DataFrame(tr)


def train_batch(b, optimizer, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    optimizer.zero_grad()
    with autocast():
        result = model[b]
        loss_class = model.obj_loss(result) # refine_obj_cls 为 False 情况下默认返回 0
        loss_rel = model.rel_loss(result)
        loss_scpred = model.scpred_loss(result)

        loss = loss_class + loss_rel + loss_scpred # 成本函数
    with amp.scale_loss(loss, optimizer) as scaled_loss: # 损失缩放，混合精度
        scaled_loss.backward() # 启用反向传播，计算出各个参数的梯度
    clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.grad is not None],  # 所有叶子节点，即 W、B
                   max_norm=conf.clip, verbose=verbose, clip=True) # 梯度裁剪，所有参数梯度 L2 范数的和不能超过 conf.clip
    optimizer.step() # 梯度下降
    return result, {
        'loss_class': float(loss_class),
        'loss_rel': float(loss_rel),
        'loss_scpred': float(loss_scpred),
        'loss_total': float(loss),
    }
