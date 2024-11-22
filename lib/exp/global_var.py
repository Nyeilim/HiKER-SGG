import os

import numpy as np
import torch
from tqdm import tqdm

from config import ModelConfig, ALL_EDGE, NODE_EMBEDDING, REL_COUNTS, CONF_MAT_FREQ_TRAIN, CONF_MAT_UPDATED
from dataloaders.visual_genome import VGDataLoader, VG
from lib.my_model_24 import KERN
from lib.my_util import adj_normalize
from lib.pytorch_misc import optimistic_restore

# ------------------------------------------------------------------------------------

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
-nwork 24
-ngpu 1
-lr 1e-4
-nepoch 20
-pooling_dim 4096
-ggnn_rel_time_step_num 3
-ggnn_rel_hidden_dim 1024
-adam
-require_overlap_det
-use_bpl
-use_sa
-use_knowledge
-use_embedding
''')

# 这里的 split 的目的好像是用来训练
# with_clean_classifier==True 表示使用 BPL Method，该方法出自论文 SGG-G2S; return size: 16832, 5000, 26446
train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                             filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=conf.use_bpl,
                             get_state=False)

# 这里的两个集合经过 BPL 方法平衡后，会少很多头部谓词样本，在 SGG-G2S 的论文中拿来微调最后的层; return size: 2104, 5000
# 它的这个 train_loader 的大小其实反应的是 batch_sampler size，说白了就是批次个数，2104 = 16832/8
# batch_sampler 是一个可迭代的对象，它返回一系列的索引列表（batches），每个列表代表一个批次。batch_sampler 可以自定义如何从数据集中抽取批次。
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',  # rel 会传给 Blob 当构造参数
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)

# 拿到 VG-SGG-dicts.json 里的 idx_to_predicates
ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship

# ------------------------------------------------------------------------------------

# 模型本体
model = KERN(
    classes=train.ind_to_classes,
    rel_classes=train.ind_to_predicates,
    num_gpus=conf.num_gpus,
    mode=conf.mode,
    require_overlap_det=conf.require_overlap_det,
    use_resnet=conf.use_resnet,
    use_proposals=conf.use_proposals,
    pooling_dim=conf.pooling_dim,
    ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
    ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim,
    ggnn_rel_output_dim=None,
    graph_path=ALL_EDGE,
    emb_path=NODE_EMBEDDING,
    rel_counts_path=REL_COUNTS,
    use_knowledge=conf.use_knowledge,
    use_embedding=conf.use_embedding,
    refine_obj_cls=conf.refine_obj_cls,
    class_volume=1.0,
    with_clean_classifier=conf.use_bpl,
    with_transfer=conf.use_sa,
    sa=conf.use_sa,
    config=conf,
)

# Freeze the detector 冻结 Faster-RCNN 参数
for n, param in model.detector.named_parameters():
    param.requires_grad = False

# 加载模型并迁移至 GPU，这里最开始加载的其实是 GB-Net 的权重
ckpt = torch.load(conf.ckpt)
optimistic_restore(model, ckpt['state_dict'], skip_clean=False)
model = model.cuda()

# ------------------------------------------------------------------------------------

# Initialize the confusion matrix 初始化混淆矩阵
initial_conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
initial_conf_matrix[0, :] = 0.0
initial_conf_matrix[:, 0] = 0.0
initial_conf_matrix[0, 0] = 1.0
initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
initial_conf_matrix = adj_normalize(initial_conf_matrix)
np.save(CONF_MAT_UPDATED, initial_conf_matrix)  # 这个玩意是拿来算概率转移矩阵的
