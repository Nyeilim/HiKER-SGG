import gc
from typing import List, Dict, Any

import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from apex import amp

from dataloaders.visual_genome import VGDataLoader
from lib.exp.optim_fn import get_optim
from lib.exp.provider import provide_dataloader
from lib.exp.train_fn import train_batch


def finetune(model, conf):
    """
    里面写的东西应该接在 test 后
    1. 用原模型过次 train_full 数据集，记录下关系的“置信概率”
    2. 挑选“置信概率”前 10% 的关系，以此构造新的精选集，用于微调分类头
    3. 进入训练模式，微调分类头；返回微调后的模型对象
    :param model: 来自 hikersgg_predcls_test.py
    :param conf: 来自 hikersgg_predcls_test.py
    :return: 微调后的 model
    """
    print('~~~~~~ finetune start ~~~~~~')
    full_raw_rels = False
    print('finetune:full_raw_rels: {} @ {}'.format(full_raw_rels, __name__))
    assert conf.num_gpus == 1
    all_gt_rels_scores = []
    train_full, train_full_loader = provide_dataloader(conf, 'confusion_matrix_val')
    model.eval()
    for raw_sample, (idx, sample) in zip(train_full, enumerate(train_full_loader)):
        with autocast():
            boxes, objs, obj_scores, rels, pred_scores = model[sample]

        gt_rels = raw_sample['gt_relations'][:, :2]  # 拿到样本标注的关系对 <s,o>
        for gt_rel in gt_rels:
            rel_idx = None  # 样本标注的关系对，在预测结果 rels 中的下标
            for i, rel in enumerate(rels):
                if np.array_equal(gt_rel, rel):
                    rel_idx = i
                    break
            assert rel_idx is not None
            score = pred_scores[rel_idx]  # 标注关系对于 51 个谓词的所有置信概率
            score_max = score[1:].max()  # 拿到预测的最大置信概率
            all_gt_rels_scores.append((idx, gt_rel, score_max))

    # 循环结束后，我们会拿到所有标注样本的置信概率，按照置信概率降序排序，取前 10%；然后按照图片索引升序排序
    need = len(all_gt_rels_scores) // 10
    all_gt_rels_scores = sorted(all_gt_rels_scores, key=lambda item: item[2], reverse=True)[:need]
    all_gt_rels_scores = sorted(all_gt_rels_scores, key=lambda item: item[0])

    # 手动构造精选集 finetune_set
    finetune_set = FinetuneSet()
    img_idxes = {item[0] for item in all_gt_rels_scores}
    print("finetune set count: {}".format(len(img_idxes)))
    for idx in img_idxes:
        img_gt_rels_scores = [row for row in all_gt_rels_scores if row[0] == idx]  # 筛选出属于某个 idx 的所有关系分数
        img_gt_rels = {tuple(item[1]) for item in img_gt_rels_scores}  # 二元组 <s, o>

        raw_image = train_full[idx]  # 原始图片数据
        raw_gt_rels = raw_image['gt_relations']  # 原始关系（三元组），shape(num_rels, 3)
        filtered_gt_rels = []  # 过滤后的三元组
        for raw_gt_rel in raw_gt_rels:
            if tuple(raw_gt_rel[:2]) in img_gt_rels:
                filtered_gt_rels.append(raw_gt_rel)
        if full_raw_rels:
            filtered_gt_rels = raw_gt_rels
        raw_image['gt_relations'] = np.array(filtered_gt_rels)  # 将过滤后的三元组转正

        # 构造对象 FinetuneImage
        image = FinetuneImage(raw_image)
        finetune_set.append(image)

    model.train()
    gc.collect()
    print('start building dataloader')

    # 利用精选集构造 DataLoader，然后进行锁住除分类头的其他参数，进行微调
    finetune_set_loader, _ = VGDataLoader.splits(
        finetune_set,
        finetune_set,
        mode='rel',  # rel 会传给 Blob 当构造参数
        batch_size=conf.batch_size,
        num_gpus=conf.num_gpus,
        num_workers=conf.num_workers
    )

    # 冻结参数
    print('start freezing model params')
    for name, param in model.named_parameters():
        if '_clean' in name:
            print("{} unfreeze.".format(name))
            continue
        param.requires_grad = False

    # 开始训练，训练一次 报错 Process finished with exit code 137
    optimizer = get_optim(model, conf)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    for batch_idx, batch in enumerate(finetune_set_loader):
        train_batch(model, conf, batch, optimizer)

    print('~~~~~~ finetune finish ~~~~~~')
    return model  # 返回训练后的模型


class FinetuneImage:

    def __init__(self, entry):
        self.img = entry['img']
        self.img_size = entry['img_size']
        self.gt_boxes = entry['gt_boxes']
        self.gt_classes = entry['gt_classes']
        self.gt_relations = entry['gt_relations']
        self.scale = entry['scale']
        self.flipped = entry['flipped']
        self.fn = entry['fn']
        self.index = entry['index']  # 在 train_full 中的原索引


class FinetuneSet(Dataset):

    def __init__(self):
        self.images: List[FinetuneImage] = []

    def __getitem__(self, index) -> Dict[str, Any]:
        entry = {
            'img': self.images[index].img,  # 放缩后的图像
            'img_size': self.images[index].img_size,  # 放缩后尺寸及放缩比例
            'gt_boxes': self.images[index].gt_boxes,  # 未放缩的 gt_box 坐标，shape(num_box, 4)
            'gt_classes': self.images[index].gt_classes,  # s,o 索引标注，shape(num_box,)
            'gt_relations': self.images[index].gt_relations,  # 关系（三元组），shape(num_rels, 3)
            'scale': self.images[index].scale,  # gt_box 放缩比例，具体的放缩逻辑在 Blob 中
            'index': index,  # 索引下标，第几张可用图片
            'flipped': self.images[index].flipped,  # 翻转标记
            'fn': self.images[index].fn,  # 文件名
        }

        return entry

    def __len__(self) -> int:
        return len(self.images)

    def append(self, sample: FinetuneImage):
        self.images.append(sample)
