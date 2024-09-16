import numpy as np
import os
import sys
import torch
from tqdm import tqdm

from config import ModelConfig
from dataloaders.visual_genome import VGDataLoader, VG
from lib.my_model_24 import KERN
from lib.my_util import adj_normalize
from lib.pytorch_misc import optimistic_restore

# ------------------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择显卡
codebase = '/output/HiKER-SGG/'  # 项目根目录
sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
exp_name = 'hikersgg_predcls_train'
write = tqdm.write  # 函数引用赋值，用来打印日志
use_bpl = True # 启用还是关闭 BPL 方法
use_sa = False # 启用还是关闭 SA 方法

# 创建配置类，加载配置
# vgrel-11 是 GB-Net 提供的预训练模型，HiKER-SGG 的核心部分(GNN)和 GB-Net 非常接近
conf = ModelConfig(f'''
-m predcls -p 2500 -clip 5
-tb_log_dir ../data/summaries/kern_predcls/{exp_name}
-save_dir ../data/checkpoints/kern_predcls/{exp_name}
-ckpt ../data/checkpoints/vgdet/vgrel-11.tar
-val_size 5000
-adam
-b 8
-nwork 24
-ngpu 1
-lr 1e-4
''')

# 修改部分配置，这个 .MODEL 是个 Munch 对象实例，Munch 类的行为逻辑就好像字典，不过它是以 .属性名 访问值
# LRGA 不知道是什么
conf.MODEL.CONF_MAT_FREQ_TRAIN = '/output/data/misc/conf_mat_freq_train.npy'  # modified
conf.MODEL.LRGA.USE_LRGA = False
conf.MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
conf.MODEL.NORMALIZE_EOA = False
# conf.MODEL.LRGA.K = 50
# conf.MODEL.LRGA.DROPOUT = 0.5
# conf.MODEL.GN.NUM_GROUPS = 1024//8

# ------------------------------------------------------------------------------------

# VG 类继承自 Dataset 类，把数据集拆分为训练集、验证集、测试集，参数作为关键字参数传入
# take train_full for evaluating the confusion matrix; return size: 57723, 5000, 26446
train_full, _val, _test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                                    use_proposals=conf.use_proposals,
                                    filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=False,
                                    get_state=False)

# VGDataLoader 类继承自 Dataloader 类，作为迭代器拿取 batch; return size: 7215, 57723
_, train_full_loader = VGDataLoader.splits(train_full, train_full, mode='rel',
                                           batch_size=conf.batch_size,
                                           num_workers=conf.num_workers,
                                           num_gpus=conf.num_gpus,
                                           pin_memory=True)

# ------------------------------------------------------------------------------------

# 这里的 split 的目的好像是用来训练
# with_clean_classifier==True 表示使用 BPL Method，该方法出自论文 SGG-G2S; return size: 16832, 5000, 26446
train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                             filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=use_bpl,
                             get_state=False)

# 这里的两个集合经过 BPL 方法平衡后，会少很多头部谓词样本，在 SGG-G2S 的论文中拿来微调最后的层; return size: 2104, 5000
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)

# 拿到 VG-SGG-dicts.json 里的 idx_to_predicates
ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship

# ------------------------------------------------------------------------------------

# 模型本体
detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                # 存储着 hierarchical knowledge graphs 中的各种边，ent2ent, pred2pred, ent2pred, pred2ent
                graph_path=os.path.join(codebase, 'graphs/005/all_edges_with_sccluster2_pred_ent.pkl'),
                # 存储着各个 ent, pred 节点的 embedding
                emb_path=os.path.join(codebase, 'graphs/001/emb_mtx_with_sccluster2_pred_ent.pkl'),
                # 存储着训练集中每个谓词的词频
                rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                class_volume=1.0, with_clean_classifier=use_bpl, with_transfer=use_sa, sa=use_sa, config=conf,
                )

# Freeze the detector 冻结参数
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

# 加载模型并迁移至 GPU，这里最开始加载的其实是 GB-Net 的权重
ckpt = torch.load(conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'], skip_clean=False)
detector.cuda()

# ------------------------------------------------------------------------------------

# Initialize the confusion matrix 初始化混淆矩阵
initial_conf_matrix = np.load(conf.MODEL.CONF_MAT_FREQ_TRAIN)
initial_conf_matrix[0, :] = 0.0
initial_conf_matrix[:, 0] = 0.0
initial_conf_matrix[0, 0] = 1.0
initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
initial_conf_matrix = adj_normalize(initial_conf_matrix)
np.save('/output/data/misc/conf_mat_updated.npy', initial_conf_matrix)  # 这个玩意是拿来算概率转移矩阵的
