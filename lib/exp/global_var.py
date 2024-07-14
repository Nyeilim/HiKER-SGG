import os
import sys

from apex.optimizers import FusedAdam, FusedSGD
from tqdm import tqdm

from config import ModelConfig
from dataloaders.visual_genome import VGDataLoader, VG
from lib.my_model_24 import KERN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择显卡
codebase = '/output/HiKER-SGG/'  # 项目根目录
sys.path.append("/output/HiKER-SGG/")  # 添加环境变量
exp_name = 'hikersgg_predcls_train'
write = tqdm.write  # 函数引用赋值，好想是用来打印日志

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

# 修改部分配置，这个 .MODEL 是个 Munch 对象实例，Munch 类的行为逻辑就好像字典，不过他是以 .属性名 访问值
conf.MODEL.CONF_MAT_FREQ_TRAIN = '/output/data/misc/conf_mat_freq_train.npy'  # modified
conf.MODEL.LRGA.USE_LRGA = False
conf.MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
conf.MODEL.NORMALIZE_EOA = False
conf.num_workers = 9
# conf.MODEL.LRGA.K = 50
# conf.MODEL.LRGA.DROPOUT = 0.5
# conf.MODEL.GN.NUM_GROUPS = 1024//8

# ------------------------------------------------------------------------------------

# For evaluating the confusion matrix; 这里 split 的目的好像是为了拿个 train_full 用于计算混淆矩阵
# VG 类继承自 Dataset 类，把数据集拆分为训练集、验证集、测试集
train_full, _val, _test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                                    use_proposals=conf.use_proposals,
                                    filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=False,
                                    get_state=False)

# VGDataLoader 类继承自 Dataloader 类，作为迭代器拿取 batch
_, train_full_loader = VGDataLoader.splits(train_full, train_full, mode='rel',
                                           batch_size=conf.batch_size,
                                           num_workers=conf.num_workers,
                                           num_gpus=conf.num_gpus,
                                           pin_memory=True)

# ------------------------------------------------------------------------------------

# 这里的 split 的目的好像是用来训练
# 这里的 with_clean_classifier 是 True，它和一个叫 BPL 的东西相关，他在论文的 [22] 中被提到
train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                             filter_non_overlap=conf.mode == 'sgdet', with_clean_classifier=True, get_state=False)

# 这里的两个集合就是正常的 train val, 和上面的 train_full 不一样
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               pin_memory=True)

ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship

# ------------------------------------------------------------------------------------

# 模型本体
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

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False


# ------------------------------------------------------------------------------------

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


optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)
