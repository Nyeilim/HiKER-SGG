#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 选择显卡
codebase = '/output/HiKER-SGG/'         # 项目根目录
sys.path.append("/output/HiKER-SGG/")   # 添加环境变量
# sys.path.append('../../../')
# sys.path.append('../../../apex')


# In[ ]:


import torch
torch.__version__


# In[ ]:


exp_name = 'hikersgg_predcls_train'


# In[ ]:


import os
from time import time as time_time
import numpy as np
# from torch import optim
from apex import amp
import torch
import pandas as pd
from tqdm import tqdm
write = tqdm.write

from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import print_para
from dataloaders.visual_genome import VGDataLoader, VG

from lib.my_model_24 import KERN

# sg val
# import numpy
# import pyximport
# pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
#                               "include_dirs":numpy.get_include()},
#                   reload_support=True)
# then delete "script_args":["--compiler=mingw32"],


# In[ ]:

# 创建配置类，加载配置
conf = ModelConfig(f'''
-m predcls -p 2500 -clip 5
-tb_log_dir ../data/summaries/kern_predcls/{exp_name}
-save_dir ../data/checkpoints/kern_predcls/{exp_name}
-ckpt ../data/checkpoints/vgdet/vgrel-11.tar
-val_size 5000
-adam
-b 3
-ngpu 1
-lr 1e-4
''')
# lr 1e-3


# In[ ]:


# modified
# 这个 .MODEL 是个 Munch 对象实例，Munch 类的行为逻辑就好像字典，不过他是以 .属性名 访问值
conf.MODEL.CONF_MAT_FREQ_TRAIN = '/output/data/misc/conf_mat_freq_train.npy'
conf.MODEL.LRGA.USE_LRGA = False
conf.MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
conf.MODEL.NORMALIZE_EOA = False
conf.num_workers = 9
# conf.MODEL.LRGA.K = 50
# conf.MODEL.LRGA.DROPOUT = 0.5
# conf.MODEL.GN.NUM_GROUPS = 1024//8


# In[ ]:

# 获取当前工作目录
os.getcwd()


# In[ ]:


# For evaluating the confusion matrix
# VG 类继承自 Dataset 类，把数据集拆分为训练集、验证集、测试集
train_full, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                            use_proposals=conf.use_proposals,
                            filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=False, get_state=False)


# In[ ]:

# VGDataLoader 类继承自 Dataloader 类，这玩意就是个迭代器
_, train_full_loader = VGDataLoader.splits(train_full, train_full, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)


# In[ ]:


train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                            use_proposals=conf.use_proposals,
                            filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=True, get_state=False)


# In[ ]:


ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship


# In[ ]:


train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)


# In[ ]:


detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                # 这三个参数是什么？
                graph_path=os.path.join(codebase, 'graphs/005/all_edges_with_sccluster2_pred_ent.pkl'),
                emb_path=os.path.join(codebase, 'graphs/001/emb_mtx_with_sccluster2_pred_ent.pkl'),
                rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                class_volume=1.0, with_clean_classifier=True, with_transfer=True, sa=True, config=conf,
                )


# In[ ]:


# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False


# In[ ]:


from apex.optimizers import FusedAdam, FusedSGD
from torch import optim

def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if (n.startswith('roi_fmap') or 'clean' in n) and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not (n.startswith('roi_fmap') or 'clean' in n) and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = FusedAdam(params, weight_decay=conf.adamwd, lr=lr, eps=1e-3)
        # optimizer = optim.Adam(params, weight_decay=conf.adamwd, lr=lr, eps=1e-3)
    else:
        optimizer = FusedSGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)
        # optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
    #                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer #, scheduler



# In[ ]:


# Initialize the confusion matrix
from lib.my_util import adj_normalize
initial_conf_matrix = np.load(conf.MODEL.CONF_MAT_FREQ_TRAIN)
initial_conf_matrix[0, :] = 0.0
initial_conf_matrix[:, 0] = 0.0
initial_conf_matrix[0, 0] = 1.0
initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
initial_conf_matrix = adj_normalize(initial_conf_matrix)
np.save('/output/data/misc/conf_mat_updated.npy', initial_conf_matrix)


# In[ ]:


ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'], skip_clean=False)
detector.cuda();


# In[ ]:


from time import time as time_time
def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time_time()
    prog_bar = tqdm(enumerate(train_loader), total=int(len(train)/train_loader.batch_size))
    for b, batch in prog_bar:
        # print(train_batch(batch, verbose=b % (conf.print_interval*10) == 0))
        result, loss_dict = train_batch(batch, verbose=b % (conf.print_interval*10) == 0)
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


# In[ ]:


from torch.cuda.amp import autocast
def train_batch(b, verbose=False):
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
        result = detector[b]
        loss_class = detector.obj_loss(result)
        loss_rel = detector.rel_loss(result)
        loss_scpred = detector.scpred_loss(result)

        loss = loss_class + loss_rel + loss_scpred
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    optimizer.step()
    return result, {
          'loss_class': float(loss_class),
          'loss_rel': float(loss_rel),
          'loss_scpred': float(loss_scpred),
          'loss_total': float(loss),
    }


# In[ ]:


from torch import no_grad as torch_no_grad
from tqdm import tqdm

def val_epoch():
    detector.eval()
    evaluator_list = [] # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes() # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    prog_bar = tqdm(enumerate(val_loader), total=int(len(val)/val_loader.batch_size))

    with torch_no_grad():
        for val_b, batch in prog_bar:
            val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)

    detector.train()
    return recall, recall_mp, mean_recall, mean_recall_mp

def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    with autocast():
        det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)



# In[ ]:


def train_evaluate():
    detector.eval()
    evaluator_list = [] # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes() # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)

    prog_bar = tqdm(enumerate(train_full_loader), total=int(len(train_full)/train_full_loader.batch_size))

    with torch_no_grad():
        for train_full_b, batch in prog_bar:
            train_full_batch(conf.num_gpus * train_full_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)
            if train_full_b == 10000: # For efficiency, only evaluate 10000 batches
                break
    confusion_matrix = evaluator[conf.mode].result_dict['predicate_confusion_matrix']
    detector.train()
    return confusion_matrix

def train_full_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    with autocast():
        det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': train_full.gt_classes[batch_num + i].copy(),
            'gt_relations': train_full.relationships[batch_num + i].copy(),
            'gt_boxes': train_full.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)


# In[ ]:


from warnings import warn
# torch.backends.cudnn.benchmark = True

alpha = 0.9
optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
detector, optimizer = amp.initialize(detector, optimizer, opt_level="O0")

start_epoch = 0
end_epoch = 20

conf_matrix_list = []
for epoch in range(start_epoch, end_epoch):
    if (epoch + 1) % 3 == 0:
        print('Evaluating new confusion matrix...')
        conf_matrix = train_evaluate()
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
    if epoch != 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

    rez = train_epoch(epoch)
    losses_mean_epoch = rez.mean(axis=0)
    losses_mean_epoch_class = losses_mean_epoch['loss_class']
    losses_mean_epoch_rel = losses_mean_epoch['loss_rel']
    losses_mean_epoch_total = losses_mean_epoch['loss_total']
    write("overall{:2d}: ({:.3f})\n{}".format(epoch, losses_mean_epoch_total, losses_mean_epoch))

    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))
        # noinspection PyPackageRequirements
        print(os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()

