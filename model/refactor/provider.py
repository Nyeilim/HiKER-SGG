import torch

from config import ALL_EDGE, NODE_EMBEDDING, REL_COUNTS, DATALOADER_MODES
from model.dataloaders.visual_genome import VG, VGDataLoader
from model.hiker_model import HiKER
from model.pytorch_misc import optimistic_restore


def provide_dataloader(conf, dataloader_mode):
    assert dataloader_mode in DATALOADER_MODES

    train, val, test = VG.splits(
        num_val_im=conf.val_size,
        filter_duplicate_rels=conf.filter_duplicate_rels,
        use_proposals=conf.use_proposals,
        filter_non_overlap=conf.mode == 'sgdet',
        with_clean_classifier=conf.use_bpl if dataloader_mode == 'train' else False,
    )

    if dataloader_mode == 'test':
        val = test # 复用 val_epoch 进行测试
    elif dataloader_mode == 'confusion_matrix_val':
        val = train

    train_loader, val_loader = VGDataLoader.splits(
        train,
        val,
        mode='rel',  # rel 会传给 Blob 当构造参数
        batch_size=conf.batch_size,
        num_gpus=conf.num_gpus,
        num_workers=conf.num_workers
    )

    if dataloader_mode == 'train':
        return train, train_loader
    else:
        return val, val_loader


def provide_model(conf, ind_to_classes, ind_to_predicates):
    # 模型本体
    model = HiKER(
        classes=ind_to_classes,
        rel_classes=ind_to_predicates,
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

    return model
