"""
HiKER-SGG 训练和测试工具函数

任务配置差异总结：

PredCls 任务：
- 训练：从 GB-Net 预训练模型 (vgrel-11.tar) 开始，学习率 1e-4，无额外标志
- 测试：使用 PredCls 训练的最佳模型，学习率 1e-4，无额外标志
- 混淆矩阵：从初始混淆矩阵开始训练

SGCls 任务：
- 训练：从 PredCls 训练好的最佳模型开始，学习率 1e-5，添加 -refine_obj_cls 标志
- 测试：使用 SGCls 训练的最佳模型，学习率 1e-5，添加 -refine_obj_cls 标志  
- 混淆矩阵：从 PredCls 训练好的混淆矩阵开始训练
"""

import os
import sys

import numpy as np
import torch
from apex import amp
from tqdm import tqdm
import pyximport

sys.path.append("/output/HiKER-SGG/")  # 添加环境变量

# 这行语句会自动编译项目里 .pyx 文件，这里会导致两次编译，分别是 box_intersections_cpu 和 draw_rectangles
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import config
from model.refactor.val_fn import val_epoch, confusion_matrix_evaluate
from model.refactor.provider import provide_model, provide_dataloader
from model.refactor.util import save_best_matrices, load_best_matrices, cleanup_model_files
from model.refactor.optim_fn import get_optim
from model.refactor.train_fn import train_epoch
from model.util import adj_normalize
from config import data_path, CONF_MAT_UPDATED, CONF_MAT_FREQ_TRAIN, ModelConfig, ALPHA, print_globals

# 设置GPU环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
write = tqdm.write  # 函数引用赋值，用来打印日志


def get_predcls_checkpoint(task_type='predcls'):
    """
    获取PredCls任务的checkpoint路径
    :param task_type: 任务类型
    :return: checkpoint路径
    """
    if task_type == 'predcls':
        return 'checkpoints/vgdet/vgrel-11.tar'  # GB-Net预训练模型
    elif task_type == 'sgcls':
        # SGCls需要从PredCls训练好的模型开始
        try:
            predcls_best_epoch = load_best_matrices('predcls')['best_mr_epoch']
            print("Using PredCls best model from epoch {}".format(predcls_best_epoch))
        except:
            predcls_best_epoch = 10
            print("Using default PredCls model from epoch {}".format(predcls_best_epoch))
        return f'checkpoints/kern_predcls/hikersgg_predcls_train/vgrel-{predcls_best_epoch}.tar'
    else:
        raise ValueError("Unsupported task type: {}".format(task_type))


def get_learning_rate(task_type='predcls'):
    """
    根据任务类型获取学习率
    :param task_type: 任务类型
    :return: 学习率
    """
    if task_type == 'predcls':
        return '1e-4'
    elif task_type == 'sgcls':
        return '1e-5'  # SGCls通常需要更小的学习率
    else:
        raise ValueError("Unsupported task type: {}".format(task_type))


def get_task_specific_flags(task_type='predcls'):
    """
    根据任务类型获取特定的配置标志
    :param task_type: 任务类型
    :return: 配置标志字符串
    """
    if task_type == 'predcls':
        return ''
    elif task_type == 'sgcls':
        return '-refine_obj_cls'
    else:
        raise ValueError("Unsupported task type: {}".format(task_type))


def create_train_config(task_type='predcls', exp_name=None):
    """
    创建训练配置
    :param task_type: 任务类型
    :param exp_name: 实验名称
    :return: ModelConfig对象
    """
    if exp_name is None:
        exp_name = f'hikersgg_{task_type}_train'
    
    checkpoint = get_predcls_checkpoint(task_type)
    learning_rate = get_learning_rate(task_type)
    task_flags = get_task_specific_flags(task_type)
    
    config_str = f'''
	-m {task_type}
	-p 2500
	-clip 5
	-tb_log_dir summaries/kern_{task_type}/{exp_name}
	-save_dir checkpoints/kern_{task_type}/{exp_name}
	-ckpt {checkpoint}
	-val_size 5000
	-b 8
	-nwork 8
	-ngpu 1
	-lr {learning_rate}
	-nepoch 15
	-pooling_dim 4096
	-ggnn_rel_time_step_num 3
	-ggnn_rel_hidden_dim 1024
	-adam
	-require_overlap_det
	-use_bpl
	-use_knowledge
	-use_embedding
	-filter_duplicate_rels
	{task_flags}'''
    
    return ModelConfig(config_str)


def create_test_config(task_type='predcls', epoch=None):
    """
    创建测试配置
    :param task_type: 任务类型
    :param epoch: 要测试的epoch
    :return: ModelConfig对象
    """
    if epoch is None:
        best_matrices = load_best_matrices(task_type)
        epoch = best_matrices['best_mr_epoch']
    
    learning_rate = get_learning_rate(task_type)
    task_flags = get_task_specific_flags(task_type)
    
    config_str = f'''
	-m {task_type}
	-p 2500
	-clip 5
	-ckpt checkpoints/kern_{task_type}/hikersgg_{task_type}_train/vgrel-{epoch}.tar
	-b 8
	-nwork 8
	-ngpu 1
	-lr {learning_rate}
	-pooling_dim 4096
	-ggnn_rel_time_step_num 3
	-ggnn_rel_hidden_dim 1024
	-test
	-require_overlap_det
	-use_bpl
	-use_knowledge
	-use_embedding
	-filter_duplicate_rels
	{task_flags}'''
    
    return ModelConfig(config_str)


def initialize_confusion_matrix(task_type='predcls'):
    """
    初始化混淆矩阵
    :param task_type: 任务类型
    """
    if task_type == 'predcls':
        # PredCls任务：初始化混淆矩阵
        initial_conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
        initial_conf_matrix[0, :] = 0.0
        initial_conf_matrix[:, 0] = 0.0
        initial_conf_matrix[0, 0] = 1.0
        initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
        initial_conf_matrix = adj_normalize(initial_conf_matrix)
        np.save(CONF_MAT_UPDATED, initial_conf_matrix)
    elif task_type == 'sgcls':
        # SGCls任务：从PredCls训练好的混淆矩阵开始
        try:
            predcls_best_epoch = load_best_matrices('predcls')['best_mr_epoch']
            matrix_suffix = predcls_best_epoch - (predcls_best_epoch + 1) % 3
            if matrix_suffix < 2:
                conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
            else:
                conf_matrix = np.load(data_path('misc/conf/conf_mat_updated_{}.npy'.format(matrix_suffix)))
            np.save(CONF_MAT_UPDATED, conf_matrix)
        except:
            # 如果无法获取，使用默认的混淆矩阵
            initial_conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
            initial_conf_matrix[0, :] = 0.0
            initial_conf_matrix[:, 0] = 0.0
            initial_conf_matrix[0, 0] = 1.0
            initial_conf_matrix = initial_conf_matrix / (initial_conf_matrix.sum(-1)[:, None] + 1e-8)
            initial_conf_matrix = adj_normalize(initial_conf_matrix)
            np.save(CONF_MAT_UPDATED, initial_conf_matrix)


def setup_confusion_matrix_for_test(epoch, task_type='predcls'):
    """
    为测试设置混淆矩阵
    :param epoch: 要测试的epoch
    :param task_type: 任务类型
    """
    matrix_suffix = epoch - (epoch + 1) % 3  # 混淆矩阵会每三轮计算一次
    if matrix_suffix < 2:
        conf_matrix = np.load(CONF_MAT_FREQ_TRAIN)
    else:
        conf_matrix = np.load(data_path('misc/conf/conf_mat_updated_{}.npy'.format(matrix_suffix)))
    np.save(CONF_MAT_UPDATED, conf_matrix)


def print_loss_info(epoch, losses_mean_epoch, task_type='predcls'):
    """
    打印损失信息
    :param epoch: 当前epoch
    :param losses_mean_epoch: 平均损失
    :param task_type: 任务类型
    """
    losses_mean_epoch_class = losses_mean_epoch['loss_class']
    losses_mean_epoch_rel = losses_mean_epoch['loss_rel']
    losses_mean_epoch_total = losses_mean_epoch['loss_total']
    
    # SGCls任务可能还有额外的损失函数
    if task_type == 'sgcls' and 'loss_scent' in losses_mean_epoch:
        losses_mean_epoch_scent = losses_mean_epoch['loss_scent']
        write("overall{:2d}: ({:.3f}) [class:{:.3f}, rel:{:.3f}, scent:{:.3f}]\n{}".format(
            epoch, losses_mean_epoch_total, losses_mean_epoch_class, losses_mean_epoch_rel, 
            losses_mean_epoch_scent, losses_mean_epoch))
    else:
        write("overall{:2d}: ({:.3f})\n{}".format(epoch, losses_mean_epoch_total, losses_mean_epoch))


def train_flow(task_type='predcls', exp_name=None):
    """
    训练流程
    :param task_type: 任务类型 ('predcls' 或 'sgcls')
    :param exp_name: 实验名称
    """
    print("Starting {} training process...".format(task_type))
    
    # 创建配置
    conf = create_train_config(task_type, exp_name)
    conf.print_self_config()
    print_globals(config)
    
    # 初始化混淆矩阵
    initialize_confusion_matrix(task_type)
    
    # 创建数据加载器和模型
    train_set, train_set_loader = provide_dataloader(conf, 'train')
    val_set, val_set_loader = provide_dataloader(conf, 'val')
    matrix_val_set, matrix_val_set_loader = provide_dataloader(conf, 'confusion_matrix_val')
    model = provide_model(conf, train_set.ind_to_classes, train_set.ind_to_predicates)
    optimizer = get_optim(model, conf)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    
    # 初始化列表
    conf_matrix_list = []
    matrices_list = []  # mean recall of each epoch
    nc_matrices_list = []  # mean recall without constraint of each epoch
    
    # 开始训练
    for epoch in range(conf.num_epochs):
        if (epoch + 1) % 3 == 0:  # 每三轮重新计算一次混淆矩阵
            print('Evaluating new confusion matrix...')
            # 获取新的谓词混淆矩阵
            conf_matrix = confusion_matrix_evaluate(model, conf, matrix_val_set, matrix_val_set_loader)
            conf_matrix[0, :] = 0.0
            conf_matrix[:, 0] = 0.0
            conf_matrix[0, 0] = 1.0
            conf_matrix = conf_matrix / (conf_matrix.sum(-1)[:, None] + 1e-8)
            conf_matrix = adj_normalize(conf_matrix)  # 行归一化
            conf_matrix_list.append(conf_matrix)

            conf_matrix_old = np.load(CONF_MAT_UPDATED)  # 加载上轮 epoch 的转移概率矩阵
            conf_matrix_new = conf_matrix_old * ALPHA + conf_matrix * (1 - ALPHA)  # 对应公式 (20)
            np.save(CONF_MAT_UPDATED, conf_matrix_new)
            np.save(data_path('misc/conf/conf_mat_updated_{}.npy'.format(epoch)), conf_matrix_new)

        write('epoch = {}'.format(epoch))
        # 调整学习率
        if epoch != 0 and epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        rez = train_epoch(model, conf, train_set, train_set_loader, epoch, optimizer)  # 开始训练
        losses_mean_epoch = rez.mean(axis=0)
        print_loss_info(epoch, losses_mean_epoch, task_type)

        if conf.save_dir is not None:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, os.path.join(conf.save_dir, 'vgrel-{}.tar'.format(epoch)))
            print(os.path.join(conf.save_dir, 'vgrel-{}.tar'.format(epoch)))

        matrices = val_epoch(model, conf, val_set, val_set_loader)  # 开始评估
        matrices_list.append(matrices[2]) # mean_recall
        nc_matrices_list.append(matrices[3]) # mean_recall_mp

    # 保存最佳模型信息并清理
    save_best_matrices(matrices_list, nc_matrices_list, task_type)
    best_matrices = load_best_matrices(task_type)
    keep_epochs = [
        best_matrices['best_mr_epoch'],  # 最佳模型
        best_matrices['second_best_mr_epoch']  # 第二佳模型
    ]
    print("Training completed, starting to clean up model files, keeping models from epoch {} and {}".format(keep_epochs[0], keep_epochs[1]))
    cleanup_model_files(task_type, keep_epochs=keep_epochs)


def test_flow(task_type='predcls'):
    """
    测试流程
    :param task_type: 任务类型 ('predcls' 或 'sgcls')
    """
    print("Starting {} testing process...".format(task_type))
    print_globals(config)

    # 获取前两个最佳模型的epoch
    best_matrices = load_best_matrices(task_type)
    test_epochs = [
        best_matrices['best_mr_epoch'],  # 最佳模型
        best_matrices['second_best_mr_epoch']  # 第二佳模型
    ]
    print("Will test the top two best models: epoch {} and epoch {}".format(test_epochs[0], test_epochs[1]))
    
    # 测试每个模型
    for epoch in test_epochs:
        print("Starting to test model from epoch {}...".format(epoch))
        
        # 创建配置
        conf = create_test_config(task_type, epoch)
        conf.print_self_config()
        
        # 设置混淆矩阵
        setup_confusion_matrix_for_test(epoch, task_type)
        
        # 加载数据集和模型
        test_set, test_set_loader = provide_dataloader(conf, 'test')
        model = provide_model(conf, test_set.ind_to_classes, test_set.ind_to_predicates)
        
        # 开始评估
        val_epoch(model, conf, test_set, test_set_loader) 