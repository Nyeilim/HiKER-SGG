#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
codebase = '/output/HiKER-SGG/'
sys.path.append("/output/HiKER-SGG/")
# sys.path.append('../../../')
# sys.path.append('../../../apex')


# In[ ]:


import torch
torch.__version__


# In[ ]:


exp_name = 'hikersgg_predcls_test' # Change to the experiment name


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


# Change ckpt path for the evaluated model
# If you want to test on VG-C benchmark, include "-test_n" in the command line
conf = ModelConfig(f'''
-m predcls -p 2500 -clip 5
-ckpt ../data/checkpoints/kern_predcls/hikersgg_predcls_train/vgrel-10.tar
-b 3
-ngpu 1
-test
-lr 1e-4
''')

# Also load the corresponding confusion matrix
conf_matrix = np.load('/output/data/misc/conf/conf_mat_updated_8.npy') # Remember to change to the path of the confusion matrix
np.save('/output/data/misc/conf_mat_updated.npy', conf_matrix)


# In[ ]:


# modified
conf.MODEL.CONF_MAT_FREQ_TRAIN = '/output/data/misc/conf_mat_freq_train.npy'
conf.MODEL.LRGA.USE_LRGA = False
conf.MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
conf.MODEL.NORMALIZE_EOA = False
conf.num_workers = 9
# conf.MODEL.LRGA.K = 50
# conf.MODEL.LRGA.DROPOUT = 0.5
# conf.MODEL.GN.NUM_GROUPS = 1024//8


# In[ ]:


os.getcwd()


# In[ ]:


train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                            use_proposals=conf.use_proposals,
                            filter_non_overlap=conf.mode == 'sgdet',
                            with_clean_classifier=False,
                            get_state=False, test_n=conf.test_n)


# In[ ]:


ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship
# Evaluate on test set
# Here we let val = test since we want to call val_epoch() for evaluation
if conf.test or conf.test_n:
    val = test


# In[ ]:


_, train_loader = VGDataLoader.splits(train, train, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)


# In[ ]:


_, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)


# In[ ]:


detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                graph_path=os.path.join(codebase, 'graphs/005/all_edges_with_sccluster2_pred_ent.pkl'),
                emb_path=os.path.join(codebase, 'graphs/001/emb_mtx_with_sccluster2_pred_ent.pkl'),
                rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                class_volume=1.0, with_clean_classifier=True, with_transfer=True, sa=True, config=conf,
               )


# In[ ]:


ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'], skip_clean=False)
detector.cuda();


# In[ ]:


print(print_para(detector), flush=True)


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


# In[ ]:


from torch.cuda.amp import autocast

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


detector.eval()

recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()

