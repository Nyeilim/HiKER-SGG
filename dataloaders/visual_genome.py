"""
File that involves dataloaders for the Visual Genome dataset.
"""
import json
import os
import sys
import numpy as np
from os import environ as os_environ
from collections import defaultdict

from h5py import File as h5py_File
from os.path import join as os_path_join, exists as os_path_exists
from json import load as json_load
from numpy import array as np_array, where as np_where, \
    zeros_like as np_zeros_like, all as np_all, column_stack as np_column_stack, \
    zeros as np_zeros, int32 as np_int32
from numpy.random import random as np_random_random, choice as np_random_choice
from PIL import Image
from PIL.Image import open as Image_open, FLIP_LEFT_RIGHT as Image_FLIP_LEFT_RIGHT, LANCZOS as Image_LANCZOS
from torch.utils.data import Dataset, DataLoader
from torch import save as torch_save, load as torch_load
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from pycocotools.coco import COCO
from dataloaders.blob import Blob
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN, BPL_LIMIT, \
    BPL_TOPK_NUM
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from PIL import ImageDraw
from .corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, zoom_blur, motion_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate, sunglare, waterdrop, wildfire_smoke, rain, dust
import matplotlib.pyplot as plt


class VG(Dataset):
    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=VG_SGG_DICT_FN,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 use_proposals=False, with_clean_classifier=None, get_state=False, caching=False, use_cache=False, test_n=False):
        """
        Torch dataset for VisualGenome
        :param mode: Must be `train`, `test`, or `val`
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        self.use_cache = use_cache
        self.caching = caching
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode
        self.test_n = test_n

        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'

        # BPL 的改进项
        self.non_rel_revise = False # 空关系修正
        self.random_balanced_sample = True # 随机平衡采样

        # 这个 dict_file 就是 VG-SGG-dicts.json
        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)  # contiguous 151, 51 containing __background__
        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap and self.is_train,
            dict_file=dict_file,
            with_clean_classifier=with_clean_classifier,
            ind_to_predicates=self.ind_to_predicates,
            non_rel_revise=self.non_rel_revise,
            random_balanced_sample=self.random_balanced_sample
        )

        self.filenames = load_image_filenames(image_file)
        self.filenames = [self.filenames[i] for i in np_where(self.split_mask)[0]]  # 把“可用图片”的名字挑出来

        if use_proposals:
            print("Loading proposals", flush=True)
            with h5py_File(PROPOSAL_FN, 'r') as p_h5:
                rpn_rois = p_h5['rpn_rois']
                rpn_scores = p_h5['rpn_scores']
                rpn_im_to_roi_idx = np_array(p_h5['im_to_roi_idx'][self.split_mask])
                rpn_num_rois = np_array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np_column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]))

        # 这个是图片放缩的处理流程，会在 __getitem__ 里面使用
        tform = [
            SquarePad(),    # 进行图像边缘填充，使图像变为正方形
            Resize(IM_SCALE),   # 默认为 592, 说明图片将被放缩到 592x592
            ToTensor(),     # 将图像数据转换为 PyTorch 张量格式
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # 将张量归一化
        ]
        self.transform_pipeline = Compose(tform)    # 将流程封装成函数

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    # 类方法，就是 Java 里面那种使用类名调用的方法， cls 参数代表类方法的那个类，此处代表 VG 类
    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)   # 这段就相当于调用 VG 类的构造函数
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    # 这个方法是 Dataset 的抽象方法，必须得实现这个方法
    def __getitem__(self, index):
        fname = self.filenames[index]
        cache_path = os_path_join(f'cached_{self.mode}', f'{fname}.pt')
        if self.caching is True and (self.use_cache is True or os_path_exists(cache_path)):
            return torch_load(cache_path)

        # 按照文件名读取照片，以 RGB 像素格式读取为二进制数据
        image_unpadded = Image_open(fname).convert('RGB')
        w, h = image_unpadded.size
        max_side = max(w, h)    # 取长边

        # 似乎是对图片后处理，添加损坏(corruptions)的逻辑
        if self.test_n:
            ################### Apply corruptions to the image ####################
            # image_unpadded = gaussian_noise(image_unpadded, severity=5)
            # image_unpadded = shot_noise(image_unpadded, severity=5)
            # image_unpadded = impulse_noise(image_unpadded, severity=5)
            # image_unpadded = defocus_blur(image_unpadded, severity=5)
            # image_unpadded = glass_blur(image_unpadded, severity=5)
            # image_unpadded = motion_blur(image_unpadded, severity=5)
            # image_unpadded = zoom_blur(image_unpadded, severity=5)
            # image_unpadded = snow(image_unpadded, severity=5)
            # image_unpadded = frost(image_unpadded, severity=5)
            # image_unpadded = fog(image_unpadded, severity=5)
            # image_unpadded = brightness(image_unpadded, severity=5)
            # image_unpadded = contrast(image_unpadded, severity=5)
            # image_unpadded = jpeg_compression(image_unpadded, severity=5)
            # image_unpadded = pixelate(image_unpadded, severity=5)
            # image_unpadded = elastic_transform(image_unpadded, severity=5)
            # image_unpadded = sunglare(image_unpadded, severity=5)
            # image_unpadded = waterdrop(image_unpadded, severity=5)
            # image_unpadded = wildfire_smoke(image_unpadded, severity=5)
            # image_unpadded = rain(image_unpadded, severity=5)
            # image_unpadded = dust(image_unpadded, severity=5)

            image_unpadded = Image.fromarray(image_unpadded.astype(np.uint8))   # 将二进制数组转换为 PIL 图像对象

            # For debugging
            # print(image_unpadded.size)
            # display(image_unpadded)
            # sys.exit()

            ################### To visualize the corrupted images ###################
            # if index == 90:
            #     correpted_list = []
            #     image_unpadded1 = gaussian_noise(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = shot_noise(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = impulse_noise(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = defocus_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = glass_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = motion_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = zoom_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = snow(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = frost(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = fog(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = brightness(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = contrast(image_unpadded, severity=1)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = elastic_transform(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = pixelate(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = jpeg_compression(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = sunglare(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = waterdrop(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = wildfire_smoke(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = rain(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = dust(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))

            #     # Show all the corrupted images in a single 4x5 plot
            #     fig, axs = plt.subplots(4, 5, figsize=(20, 12))
            #     axs = axs.flatten()
            #     name = ['Gaussian Noise', 'Shot Noise', 'Impulse Noise', 'Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur', 'Snow', 'Frost', 'Fog', 'Brightness', 'Contrast', 'Elastic Transform', 'Pixelate', 'JPEG Compression', 'Sunlight glare', 'Waterdrop', 'Wildfire Smoke', 'Rain', 'Dust']
            #     for img, ax in zip(correpted_list, axs):
            #         ax.imshow(img)
            #         ax.axis('off')
            #         ax.set_title(name[correpted_list.index(img)], fontsize=14)
            #     plt.tight_layout()
            #     plt.show()
            #     sys.exit()
            ##########################################################################

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np_random_random() > 0.5    # 翻转标记，有 50% 的概率翻转图像
        gt_boxes = self.gt_boxes[index].copy()

        box_scale_factor = BOX_SCALE / max_side # 计算放缩因子，因为待会要对图像放缩，所以 bbox 也得放缩
        # Boxes are already at BOX_SCALE
        if self.is_train:

            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                None, box_scale_factor * h)
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                None, box_scale_factor * w)

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        # 翻转图像
        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image_FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max_side
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        gt_rels = self.relationships[index].copy()  # 获取关系（三元组）
        # 使用 Set 过滤掉重复关系
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np_random_choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np_array(gt_rels)

        # 封装最后返回的数据结构
        entry = {
            'img': self.transform_pipeline(image_unpadded), # 放缩后的图像
            'img_size': im_size,    # 放缩后尺寸及放缩比例
            'gt_boxes': gt_boxes,   # 未放缩的 gt_box 坐标，shape(num_box, 4)
            'gt_classes': self.gt_classes[index].copy(),    # s,o 索引标注，shape(num_box,)
            'gt_relations': gt_rels,    # 关系（三元组），shape(num_rels, 3)
            'scale': IM_SCALE / BOX_SCALE,  # gt_box 放缩比例，具体的放缩逻辑在 Blob 中
            'index': index, # 索引下标，第几张可用图片
            'flipped': flipped, # 翻转标记
            'fn': fname,    # 文件名
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        if self.caching is True: torch_save(entry, cache_path)
        # self.cached_data.append(entry)
        return entry

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json_load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os_path_join(image_dir, basename)
        if os_path_exists(filename):
            fns.append(filename)
    assert len(fns) == 108073
    return fns


def load_graphs(
        graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
        filter_non_overlap=False, dict_file=None, with_clean_classifier=None, ind_to_predicates=None,
        non_rel_revise = False, random_balanced_sample = False
):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             split_mask: numpy boolean array of length 108073 tells you whether every image in dataset is selected.
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    # 这里的 graphs_file 就是数据集的二进制标注文件
    with h5py_File(graphs_file, 'r') as roi_h5:
        data_split = roi_h5['split'][:] # 长度为 108073(75651+32422) 的数组，每个元素为 0(代表训练集) 或者 2(代表测试集)
        split = 2 if mode == 'test' else 0
        split_mask = data_split == split # 长度为 108073 的数组，每个位置为 True 或者 False

        # 过滤(filter out)图片，过滤结果：62723/75651，26446/32422
        split_mask &= roi_h5['img_to_first_box'][:] >= 0 # 没有 bbox 的图片，这项会被标记为 -1
        if filter_empty_rels:
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0 # 没有 rel 的图片，这项会被标记为 -1

        image_index = np_where(split_mask)[0] # 拿到筛选完毕的图片对应下标；np_where 的返回值类似 ([],)；因此我们要拿到元组内的数组
        # 根据设置再决定取多少张图片，把可用图片的下标拿到
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if mode == 'val':
                image_index = image_index[:num_val_im]
            elif mode == 'train':
                image_index = image_index[num_val_im:]

        # 重新初始化 mask 标记，然后把可用图片的下标对应的元素标记成 True
        split_mask = np_zeros_like(data_split, dtype=bool)
        split_mask[image_index] = True

        # Get box information
        # 数据集中共有 1145398 个 bbox 和对应的物体标注(labels)，其中属于某张图片的 bbox 会被排列在连续的索引中
        all_labels = roi_h5['labels'][:, 0] # 会拿到 shape(1145398,) 的一维数组，形如 [136,114,...]
        # 会拿到 shape(1145398,4) 的二维数组，形如 [[511, 356, 1023, 713], [...], ...]，其中四个数字分别代表 [xc,yc,w,h]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
        assert np_all(all_boxes[:, :2] >= 0)  # sanity check; 判断中心坐标是否 >= 0
        assert np_all(all_boxes[:, 2:] > 0)  # no empty box; 判断高宽是否 > 0

        # convert from xc, yc, w, h to x1, y1, x2, y2; 将中心、高宽数据转换为左上角、右下角坐标数据
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        # 前面说到属于某张图片的 bbox, rel 会被排列在连续的索引中，这里的 first, last 其实就是来框定这个索引区间（左右闭合）的
        # 比如 roi_h5['img_to_first_box'][1] = 15, roi_h5['img_to_last_box'][1] = 21
        # 那我们就可以知道索引为 1 的图片，它的 bbox 是 all_boxes[15:21+1, :]；rel 同理
        # 至于 split_mask 其实就是我们的“可用图片”，它这里使用的是 numpy 的高级索引语法
        # 把与“可用图片”相关的信息收集起来，然后放到单独的 List 中，顺序第0张【而不是索引为0】“可用图片”的对应信息会放到 List[0]
        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels; 数据集中一共标注了 622705 个关系
        _relations = roi_h5['relationships'][:] # shape(622705, 2)，这个是三元组 <s,p,o> 中的 <s,o>，每个元素是 bbox 索引
        _relation_predicates = roi_h5['predicates'][:, 0] # shape(622705,)，这个是三元组 <s,p,o> 中的 <p>，每个元素是谓词索引

    # 上面这段都是对二进制标注文件的处理，下面这个是对数据一致性确认，确保数据能够匹配上
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # 位于外层循环的变量，将在这里收集所有数据集样本的 bbox, rel
    boxes = []
    gt_classes = []
    # gt_attributes = []
    relationships = []
    pred_topk = []
    pred_num = BPL_TOPK_NUM
    pred_count=0
    # with open('./datasets/vg/VG-SGG-dicts-with-attri-info.json','r') as f:
    # 这个加载进来的是 VG-SGG-dicts.json 文件
    with open(dict_file,'r') as f:
        vg_dict_info = json_load(f)

    predicates_tree = vg_dict_info['predicate_count'] # 拿到 VG-SGG-dicts.json 里面的 predicate_count
    # predicates_tree = json.load(open('./datasets/vg/predicate_wikipedia_count.json', 'r'))
    # 根据每个谓词的 count 数从大到小排序，最终出来个列表，每个元素都是个 map.entry，也就是元组，类似 ('on', 712409)
    predicates_sort = sorted(predicates_tree.items(), key=lambda x:x[1], reverse=True)
    # 这里大概的意思是挑选出 count 在 topk 的谓词，放到 pred_topk 里面作为列表
    for pred_i in predicates_sort:
        if pred_count >= pred_num:
            break
        pred_topk.append(str(pred_i[0])) # 取 0 就是取到 item ('on', 712409) 中的谓词 'on'
        pred_count += 1

    # 这里开始就是使用 BPL Method 的逻辑
    if with_clean_classifier:
        print('Dataloader using BPL')
        root_classes = pred_topk # 类似 ['on', 'has', 'in' ... 'wears', 'standing on', 'in front of']，就是论文说的“头部谓词”
    else:
        print('Dataloader NOT using BPL')
        root_classes = None

    root_classes_count = {}
    leaf_classes_count = {}
    all_classes_count = {}
    bpl_img_filter_out_count = 0
    # image_index 的元素内容是“可用图片”的索引【但是用不上】，索引 i 是“可用图片”的顺序号
    # 如 image_index[0] = 6526，表面第 0 张可用图片是数据集中的第 6526 张图片
    for i in range(len(image_index)):
        # 取出单张图片的信息区间
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        # 取出单张图片的 bbox 和 labels
        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        # gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        # 取出单张图片的 rel
        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            # 这里可以理解成，本来 _relations 里面装的是每个 bbox 的绝对索引，转换为对于某个图片 i_obj_start 的相对索引
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np_all(obj_idx >= 0)
            assert np_all(obj_idx < boxes_i.shape[0])
            rels = np_column_stack((obj_idx, predicates))  # shape(num_rel, 3), each row representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np_zeros((0, 3), dtype=np_int32)

        # ---------------------------------------------------------------------------
        # 外层循环在运行完上面的代码后，其实就已经完成单张图片 bbox，rel 的整理，下面分别是 重叠过滤 以及 BPL
        # ---------------------------------------------------------------------------

        # 在训练时，是否过滤掉没有重叠 bbox 的图像，不重叠的 bbox 常常被认为是没有关系的，只有 sgdet 任务才打开这个开关
        if filter_non_overlap:
            assert mode == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np_where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        # 下面这段就是 BPL 算法的内容
        if root_classes is not None and mode == 'train':
            rel_temp = []
            # 遍历此图中的每个关系，并根据其属于头部谓词还是尾部谓词，执行不同逻辑，判断将其是否加入 rel_temp
            for rel_i in rels:
                rel_i_pred = ind_to_predicates[rel_i[2]] # 将谓词索引翻译成具体的谓词，比如 31 -> 'on'
                # 这个 all_classes_count 是循环外定义的 Map<string,int> 用来存放谓词的计数
                if rel_i_pred not in all_classes_count:
                    all_classes_count[rel_i_pred] = 0
                all_classes_count[rel_i_pred] = all_classes_count[rel_i_pred] + 1

                # rel_i_pred 作为尾部谓词或者“无关系”的逻辑，谓词索引 0 表示 'no_relationship'
                if rel_i_pred not in root_classes or rel_i[2] == 0:
                    rel_i_leaf = rel_i  # 添加为尾部谓词
                    if rel_i_pred not in leaf_classes_count:
                        leaf_classes_count[rel_i_pred] = 0
                    leaf_classes_count[rel_i_pred] = leaf_classes_count[rel_i_pred] + 1 # 统计尾部谓词数
                    rel_temp.append(rel_i_leaf) # 添加该三元组（或者说关系）到临时列表，作为训练集的候选数据

                # rel_i_pred 作为头部谓词的逻辑
                if rel_i_pred in root_classes:
                    rel_i_root = rel_i
                    if rel_i_pred not in root_classes_count:
                        root_classes_count[rel_i_pred] = 0
                    # 这里人为限定：包含某个头部谓词的三元组，其样本数量不能超过 1000
                    if ((not random_balanced_sample and root_classes_count[rel_i_pred] < BPL_LIMIT)
                            or (random_balanced_sample and whether_sample(rel_i_pred, BPL_LIMIT))):
                        rel_temp.append(rel_i_root)
                        root_classes_count[rel_i_pred] = root_classes_count[rel_i_pred] + 1
                    else:
                        redundant_pred_process(rel_i_root, rel_temp, non_rel_revise)

            # 过滤仅包含多余头部谓词的图片样本
            only_root_pred = whether_only_root_pred(rel_temp, non_rel_revise)
            if only_root_pred:
                # 空关系修正的情况，实际上并不能完全过滤，需要在 vg_collate 处添加二次过滤，因为某些尾部谓词可能在后续的处理中被去掉，导致该图片仍只留下多余头部谓词
                # 可断点观察 image_index[i] == 35124 的样本，它在 Blob.append() 的信息为 'index': 9016, 'fn': '/root/VG_100K/2386175.jpg'
                split_mask[image_index[i]] = 0
                bpl_img_filter_out_count += 1
                continue
            else:
                assert not all(rel[2] == -1 for rel in rel_temp)
                rels = np_array(rel_temp, dtype=np_int32)   # 将 rel_temp 转正为 rels

        # 把此图中拿到的 bbox 和 rel 添加到总列表，开始处理下张图片
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    if with_clean_classifier and mode == 'train':
        print("~~~~~~~BPL Filter Result~~~~~~")
        print("BPL Limit: {}, BPL TopK Num: {}".format(BPL_LIMIT, BPL_TOPK_NUM))
        print("BPL:non_rel_revise: {} @ {}".format(non_rel_revise, __name__))
        print("BPL:random_balanced_sample: {} @ {}".format(random_balanced_sample, __name__))
        print("BPL:filter out images: {}".format(bpl_img_filter_out_count))
        print("origin_pred_count = {}".format(all_classes_count))
        print("root_pred_count = {}".format(root_classes_count))
        print("leaf_pred_count = {}".format(leaf_classes_count))

    return split_mask, boxes, gt_classes, relationships

def whether_sample(target_pred:str, limit:int):
    # 这个东西是我调试的时候从 all_classes_count 中弄到的
    pred_count = {
        'above': 8411, 'across': 263, 'against': 224, 'along': 493, 'and': 679, 'at': 2109, 'attached to': 1586,
        'behind': 13047, 'belonging to': 652, 'between': 511, 'carrying': 1705, 'covered in': 485, 'covering': 512,
        'eating': 702, 'flying in': 5, 'for': 1116, 'from': 198, 'growing on': 172, 'hanging from': 807, 'has': 69007,
        'holding': 11482, 'in': 24470, 'in front of': 3808, 'laying on': 779, 'looking at': 1026, 'lying on': 369,
        'made of': 128, 'mounted on': 265, 'near': 20759, 'of': 32770, 'on': 118037, 'on back of': 343, 'over': 1277,
        'painted on': 153, 'parked on': 641, 'part of': 435, 'playing': 134, 'riding': 4507, 'says': 49,
        'sitting on': 5355, 'standing on': 2496, 'to': 327, 'under': 4732, 'using': 580, 'walking in': 289,
        'walking on': 1322, 'watching': 907, 'wearing': 48582, 'wears': 4939, 'with': 12215
    }
    target_count = pred_count[target_pred]
    return np.random.random() < limit / target_count # 概率性地返回 True，这样遍历完训练集后，会有大概 limit 个该关系

def whether_only_root_pred(rel_list:list, non_rel_revise:bool):
    only_root_pred = False
    if non_rel_revise and all(rel[2] == -1 for rel in rel_list):
        only_root_pred = True
    elif not non_rel_revise and len(rel_list) == 0:
        only_root_pred = True

    return only_root_pred

def redundant_pred_process(rel_i_root, rel_temp, non_rel_revise:bool):
    # 如果没开启空关系修正，对多余的头部谓词就不用标记
    if non_rel_revise:
        # 多余的头部谓词将其标记为 -1 redundant_pred，意为冗余谓词
        rel_i_root[2] = -1
        rel_temp.append(rel_i_root)


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    with open(info_file, 'r') as f:
        info = json_load(f)
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    # 这里挺奇怪的，他直接获取文件里面的 idx_to_predicate, index_to_label 然后转成 List 不就行了
    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx'] # 这两东西是个 Map
    # 返回 List，最后的返回结果类似 ['__background__', 'airplane', 'animal', 'arm', 'bag']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates

class VGDataLoader(DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    def __iter__(self):
        iterator = super().__iter__()
        while True:
            try:
                blob = next(iterator) # 这里接收到的是 vg_collate 的返回值
                if blob is None:
                    continue  # 跳过无效的批次
                yield blob
            except StopIteration:
                break  # 捕获 StopIteration 异常并退出循环

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus, # 所有批次，后面会通过 Blob.scatter() 分发到不同 GPU
            shuffle=True,
            num_workers=num_workers,
            # 自定义的 batch 后处理函数，下面这个 lambda 表达式的入参 x 其实就是 batch
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True, # 是否丢弃最后一个不完整的批次。
            pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )
        return train_load, val_load

def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')

    # 筛选
    filtered_data = []
    for d in data:
        tmp_rels = d['gt_relations']
        if not all(rel[2] == -1 for rel in tmp_rels): # 不全为冗余谓词
            filtered_data.append(d)
    data = filtered_data
    if len(data) == 0: # 此批次无可用数据
        return None

    # 组装
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce() # 将成员变量中的各种 List，不再按照图片索引分组，而是全部堆叠成连续的 Tensor
    return blob