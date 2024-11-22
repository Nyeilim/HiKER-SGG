from dataloaders.visual_genome import VG, VGDataLoader


def provide_dataloader(conf):
    # 这里的 split 的目的好像是用来训练
    # with_clean_classifier==True 表示使用 BPL Method
    # 该方法出自论文 SGG-G2S; return size: 16832, 5000, 26446
    train, val, test = VG.splits(
        num_val_im=conf.val_size,
        filter_duplicate_rels=True,
        use_proposals=conf.use_proposals,
        filter_non_overlap=conf.mode == 'sgdet',
        with_clean_classifier=conf.use_bpl,
        get_state=False
    )

    # 这里的两个集合经过 BPL 方法平衡后，会少很多头部谓词样本，在 SGG-G2S 的论文中拿来微调最后的层; return size: 2104, 5000
    # 它的这个 train_loader 的大小其实反应的是 batch_sampler size，说白了就是批次个数，2104 = 16832/8
    # batch_sampler 是一个可迭代的对象，它返回一系列的索引列表（batches），每个列表代表一个批次。batch_sampler 可以自定义如何从数据集中抽取批次。
    train_loader, val_loader = VGDataLoader.splits(
        train,
        val,
        mode='rel',  # rel 会传给 Blob 当构造参数
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        num_gpus=conf.num_gpus,
        pin_memory=True
    )
