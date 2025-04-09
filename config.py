"""
Configuration file!
"""
import os
import logging
from argparse import ArgumentParser

from munch import Munch

ROOT_PATH = os.path.join('/output/HiKER-SGG')  # 项目文件夹根目录
METADATA_PATH = os.path.join('/root/VG_metadata')  # 斯坦福标注数据目录
VG_IMAGES = os.path.join('/root/VG_100K')  # 数据集图片目录
DATA_PATH = os.path.join('/output/data')  # 数据存储目录，大部分乱七八糟的文件都放在这


def none_and_empty(arg):
    if arg is None:  # None
        return True
    elif isinstance(arg, str) and len(arg) == 0:  # str
        return True
    else:  # int float bool others
        return False


def root_path(fn):
    return os.path.join(ROOT_PATH, fn)


def metadata_path(fn):
    return os.path.join(METADATA_PATH, fn)  # 斯坦福标注数据目录


def data_path(fn):
    return os.path.join(DATA_PATH, fn)  # 数据目录，杂七杂八的全放这


def create_data_subdir(path):
    assert path is not None and isinstance(path, str)
    if len(path.split('/')) > 1:
        filename_len = len(path.split('/')[-1])
        subdir = path[:-filename_len]
        fullpath = data_path(subdir)
        if not os.path.exists(fullpath):
            os.mkdir(fullpath)


def print_munch(obj, prefix=""):
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if not none_and_empty(value):
            if isinstance(value, Munch):
                print_munch(value, full_key)
            else:
                print(f"{full_key}: {value}")


def print_globals(module):
    print("~~~~~~~~ Global Variables in Module `{}` ~~~~~~~~".format(module.__name__))
    for name, value in module.__dict__.items():
        if name.startswith("__") or callable(value):
            continue
        if isinstance(value, Munch):
            print_munch(value, name)
            continue
        if not none_and_empty(value):
            print(f"{name}: {value}")

# =============================================================================
# 日志打印
IS_DEBUG = True
logging.basicConfig(level=logging.DEBUG if IS_DEBUG else logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG if IS_DEBUG else logging.INFO)
# =============================================================================
# 各种文件路径
## 标注元数据文件
IM_DATA_FN = metadata_path('image_data.json')
VG_SGG_FN = metadata_path('VG-SGG.h5')
VG_SGG_DICT_FN = metadata_path('VG-SGG-dicts.json')
PROPOSAL_FN = metadata_path('proposals.h5')

## 层次知识图文件
ALL_EDGE = data_path(
    'graphs/005/all_edges_with_sccluster2_pred_ent.pkl')  # 层次知识图中的各种边，ent2ent pred2pred ent2pred pred2ent
NODE_EMBEDDING = data_path('graphs/001/emb_mtx_with_sccluster2_pred_ent.pkl')  # 各个 ent pred 节点的 embedding
REL_COUNTS = data_path('graphs/001/pred_counts.pkl')  # 训练集中每个谓词的词频

## 中间数据文件
CONF_MAT_FREQ_TRAIN = data_path('misc/conf_mat_freq_train.npy')  # 初始的谓词混淆矩阵，来自 SGG-G2S 的工作，由 MotifNet 生成
CONF_MAT_UPDATED = data_path('misc/conf_mat_updated.npy')  # 途中由混淆矩阵计算出的谓词转移矩阵，用于 SA
EDGE_MATRIX = data_path('edge_matrix.npy')

## FCG
FCG_NODES = data_path('fcg_nodes.pkl')
FCG_EDGES = data_path('fcg_edges.pkl')  
FCG_METADATA = data_path('fcg_metadata.json')
# =============================================================================
# 全局变量
BPL_LIMIT = 1000 # BPL 单个谓词上限
BPL_TOPK_NUM = 15 # BPL 头部谓词数
DIS_PROGRESS_BAR = not IS_DEBUG  # 是否禁用进度条
ALPHA = 0.9 # 混淆矩阵更新权重
MODES = ('sgdet', 'sgcls', 'predcls')
DATALOADER_MODES = ('train', 'val', 'test', 'confusion_matrix_val')

# EOA code left, useless now
MODEL = Munch()
MODEL.SHIFT_EOA = False
MODEL.FOLD_EOA = False
MODEL.MERGE_EOA_SA = False
MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
MODEL.NORMALIZE_EOA = False

MODEL.LRGA = Munch()
MODEL.LRGA.USE_LRGA = False
MODEL.LRGA.K = None
MODEL.LRGA.DROPOUT = None
# =============================================================================
# 数据集中原始的 bbox 坐标使用的是 int 类型标注，它同图片像素点不是一一对应的关系，而是有个最大值 BOX_SCALE
# 比如对于一个实际宽 592 的图片，它的 x 轴标注是 900，那么我们就可以通过 900*592/1024=520.3 得到这个 x 轴标注在真实图片上的位置
BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592  # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

# IOU < thresh: negative example
RPN_POSITIVE_OVERLAP = 0.7
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25

# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256
RELS_PER_IMG_REFINE = 64
BATCHNORM_MOMENTUM = 0.01

ANCHOR_SIZE = 16
ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189)  # (0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731)  # (4, 8, 16, 32)


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    # 构造函数
    def __init__(self, args_str=None):
        """
        Defaults
        """
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.adamwd = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.cache = None
        self.mode = None
        self.test = False
        self.test_n = False
        self.adam = False
        self.use_proposals = False
        self.use_resnet = False
        self.num_epochs = None
        self.pooling_dim = None

        # self.use_ggnn_obj = False
        # self.ggnn_obj_time_step_num = None
        # self.ggnn_obj_hidden_dim = None
        # self.ggnn_obj_output_dim = None
        # self.use_obj_knowledge = False
        # self.obj_knowledge = None

        # self.use_ggnn_rel = False
        self.ggnn_rel_time_step_num = None
        self.ggnn_rel_hidden_dim = None
        # self.ggnn_rel_output_dim = None
        # self.use_rel_knowledge = False
        # self.rel_knowledge = None

        self.tb_log_dir = None
        self.save_rel_recall = None

        # 后续添加
        self.require_overlap_det = False
        self.use_bpl = False
        self.use_sa = False
        self.use_knowledge = False
        self.use_embedding = False
        self.refine_obj_cls = False
        self.filter_duplicate_rels = False

        # self.MODEL = Munch()
        # self.MODEL.DEVICE = None
        # self.MODEL.SHIFT_EOA = False
        # self.MODEL.FOLD_EOA = False
        # self.MODEL.MERGE_EOA_SA = False
        # self.MODEL.USE_ONTOLOGICAL_ADJUSTMENT = False
        # self.MODEL.NORMALIZE_EOA = False
        #
        # self.MODEL.ROI_RELATION_HEAD = Munch()
        # self.MODEL.ROI_RELATION_HEAD.BPL_HIDDEN_DIM = None
        # self.MODEL.ROI_RELATION_HEAD.BPL_POOLING_DIM = None
        # self.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER = None
        # self.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER = None
        #
        # self.MODEL.LRGA = Munch()
        # self.MODEL.LRGA.USE_LRGA = False
        # self.MODEL.LRGA.K = None
        # self.MODEL.LRGA.DROPOUT = None
        # self.MODEL.LRGA.IN_CHANNELS = None
        # self.MODEL.LRGA.HIDDEN_CHANNELS = None
        #
        # self.MODEL.GN = Munch()
        # self.MODEL.GN.NUM_GROUPS = None

        # 上面那串可以理解成显式定义，下面开始创建 parser 对象来解析参数字符串 args_str
        self.parser = self.setup_parser()
        if args_str is None:
            self.args = vars(self.parser.parse_args())
        else:
            # vars 将会返回当前类的属性字典，就是会显示每个类属性（变量）和具体的值，这里其实是将配置解析到 parser 类属性中
            # 然后再将 parser 类属性暂存到 config 类成员变量 self.args
            self.args = vars(self.parser.parse_args(args_str.split()))
        self.__dict__.update(self.args)  # 使用 parser 解析结果手动更新类属性，如果不更新就会是上面的默认的值

        self.process_path()
        self.check_param()

    def process_path(self):
        # file
        if not none_and_empty(self.ckpt):
            assert self.ckpt is not None
            create_data_subdir(self.cache)
            self.ckpt = data_path(self.ckpt)
        else:
            self.ckpt = None

        if not none_and_empty(self.cache):
            assert self.cache is not None
            create_data_subdir(self.cache)
            self.cache = data_path(self.cache)
        else:
            self.cache = None

        if not none_and_empty(self.save_rel_recall):
            assert self.save_rel_recall is not None
            create_data_subdir(self.save_rel_recall)
            self.save_rel_recall = data_path(self.save_rel_recall)
        else:
            self.save_rel_recall = None

        # dir
        if not none_and_empty(self.save_dir):
            assert self.save_dir is not None
            self.save_dir = data_path(self.save_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        else:
            self.save_dir = None

        if not none_and_empty(self.tb_log_dir):
            assert self.tb_log_dir is not None
            self.tb_log_dir = data_path(self.tb_log_dir)
            if not os.path.exists(self.tb_log_dir):
                os.makedirs(self.tb_log_dir)  # help make multi depth directories, such as summaries/kern_predcls
        else:
            self.tb_log_dir = None

    def check_param(self):
        if self.val_size < 0:
            raise ValueError("val_size should >0")

        if self.mode not in MODES:
            raise ValueError("invalid mode: mode must be in {}".format(MODES))

        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("ckpt file ({}) doesnt exist".format(self.ckpt))

    def print_self_config(self):
        print("~~~~~~~~ Manual Setting Hyperparameters ~~~~~~~~")
        for x, y in self.__dict__.items():
            if x == 'args' or x == 'parser':  # 无需打印的情况
                continue
            if isinstance(y, Munch):  # 打印非 None 子项
                print_munch(y, x)
                continue
            if not none_and_empty(y):  # 打印非 None 项
                if ((self.test == True or self.test_n == True)
                        and (x == 'val_size' or x == 'nepoch' or x == 'adam')): # Test 状态下不打印这几个值
                    continue
                print(f"{x}: {y}")

    # @formatter:off
    @staticmethod
    def setup_parser():
        """
        Sets up an argument parser
        """
        # 下面这些是默认配置
        parser = ArgumentParser(description='training code')

        parser.add_argument('-ckpt', dest='ckpt', help='filename to load from', type=str, default='')
        parser.add_argument('-save_dir', dest='save_dir', help='directory to save things to, such as checkpoints/save', default='', type=str)
        parser.add_argument('-ngpu', dest='num_gpus', help='num of gpus', type=int, default=1)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=8)
        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)
        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)
        parser.add_argument('-l2', dest='l2', help='weight decay of SGD', type=float, default=1e-4)
        parser.add_argument('-adamwd', dest='adamwd', help='weight decay of adam', type=float, default=0.0)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int, default=100)
        parser.add_argument('-m', dest='mode', help='mode in {sgdet, sgcls, predcls}', type=str, default='sgdet')
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str, default='')
        parser.add_argument('-adam', dest='adam', help='use adam', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-test_n', dest='test_n', help='test set with noise', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='num of epochs to train the model for',type=int, default=50)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='use Xu et al proposals', action='store_true')
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='dimension of pooling', type=int, default=4096)
        # parser.add_argument('-use_ggnn_obj', dest='use_ggnn_obj', help='use GGNN_obj module', action='store_true')
        # parser.add_argument('-ggnn_obj_time_step_num', dest='ggnn_obj_time_step_num', help='time step number of GGNN_obj', type=int, default=3)
        # parser.add_argument('-ggnn_obj_hidden_dim', dest='ggnn_obj_hidden_dim', help='node hidden state dimension of GGNN_obj', type=int, default=512)
        # parser.add_argument('-ggnn_obj_output_dim', dest='ggnn_obj_output_dim', help='node output feature dimension of GGNN_obj', type=int, default=512)
        # parser.add_argument('-use_obj_knowledge', dest='use_obj_knowledge', help='use object co-occurrence knowledge', action='store_true')
        # parser.add_argument('-obj_knowledge', dest='obj_knowledge', help='filename to load matrix of object co-occurrence knowledge', type=str, default='')
        # parser.add_argument('-use_ggnn_rel', dest='use_ggnn_rel', help='use GGNN_rel module', action='store_true')
        # parser.add_argument('-ggnn_rel_output_dim', dest='ggnn_rel_output_dim', help='node output feature dimension of GGNN_rel', type=int, default=512)
        # parser.add_argument('-use_rel_knowledge', dest='use_rel_knowledge', help='use co-occurrence knowledge of object pairs and relationships', action='store_true')
        # parser.add_argument('-rel_knowledge', dest='rel_knowledge', help='filename to load matrix of co-occurrence knowledge of object pairs and relationships', type=str, default='')
        parser.add_argument('-ggnn_rel_time_step_num', dest='ggnn_rel_time_step_num', help='time step number of GGNN_rel', type=int, default=3)
        parser.add_argument('-ggnn_rel_hidden_dim', dest='ggnn_rel_hidden_dim', help='node hidden state dimension of GGNN_rel', type=int, default=512)
        parser.add_argument('-tb_log_dir', dest='tb_log_dir', help='dir to save tensorboard summaries', type=str, default='')
        parser.add_argument('-save_rel_recall', dest='save_rel_recall', help='dir to save relationship results', type=str, default='')

        # 后续添加
        parser.add_argument('-require_overlap_det', dest='require_overlap_det', help='filter out not overlap pair-boxes while sgdet', action='store_true')
        parser.add_argument('-use_bpl', dest='use_bpl', help='use BPL(Balanced Predicate Learning) method to filter out training set', action='store_true')
        parser.add_argument('-use_sa', dest='use_sa', help='use SA(Semantic Adjustment) method to adjust predicate prediction', action='store_true')
        parser.add_argument('-use_knowledge', dest='use_knowledge', help='use ALL_EDGE file to initial predicate', action='store_true')
        parser.add_argument('-use_embedding', dest='use_embedding', help='use NODE_EMBEDDING to initial entity', action='store_true')
        parser.add_argument('-refine_obj_cls', dest='refine_obj_cls', help='add additional process towards entities in model', action='store_true')
        parser.add_argument('-filter_duplicate_rels', dest='filter_duplicate_rels', help='filter out duplicate rels while loading dataset', action='store_true')


        return parser
