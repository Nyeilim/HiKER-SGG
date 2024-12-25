from model.pytorch_misc import enumerate_by_image, random_choose
import torch
from model.pytorch_misc import diagonal_inds, to_variable
from config import RELS_PER_IMG, REL_FG_FRACTION


@to_variable
def proposal_assignments_gtbox(rois, gt_boxes, gt_classes, gt_rels, image_offset, add_bg_rels):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets. 这个方法会添加大量的背景关系
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]. Not needed it seems
    :param gt_classes: [num_boxes, 2] array of [img_ind, class]
        Note, the img_inds here start at image_offset
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type].
        Note, the img_inds here start at image_offset
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        bbox_targets [num_rois, 4] array of targets for the labels.
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1 ind, rel type)
    """
    im_inds = rois[:,0].long()
    labels = gt_classes[:,1].contiguous()
    rel_labels = postprocess_rels(im_inds, gt_boxes, gt_rels, image_offset, add_bg_rels)

    return rois, labels, rel_labels

def postprocess_rels(im_inds, gt_boxes, gt_rels, image_offset, add_bg_rels):
    num_im = im_inds[-1] + 1

    # Offset the image indices in fg_rels to refer to absolute indices (not just within img i)
    fg_rels = gt_rels.clone() # 前景关系
    fg_rels[:,0] -= image_offset
    offset = {}
    for i, s, e in enumerate_by_image(im_inds):
        offset[i] = s
    for i, s, e in enumerate_by_image(fg_rels[:, 0]):
        fg_rels[s:e, 1:3] += offset[i]

    # Try ALL things, not just intersections.
    is_cand = (im_inds[:, None] == im_inds[None]) # 用来表示位于同张图中的 roi 之间的潜在关系，shape(num_gt_boxes, num_gt_boxes)
    is_cand.view(-1)[diagonal_inds(is_cand)] = 0 # 去除对角线，即 roi 的自相关关系

    # # Compute salience
    # gt_inds = fg_rels[:, 1:3].contiguous().view(-1)
    # labels_arange = labels.data.new(labels.size(0))
    # torch.arange(0, labels.size(0), out=labels_arange)
    # salience_labels = ((gt_inds[:, None] == labels_arange[None]).long().sum(0) > 0).long()
    # labels = torch.stack((labels, salience_labels), 1)

    # Add in some BG labels
    is_cand.view(-1)[fg_rels[:,1]*im_inds.size(0) + fg_rels[:,2]] = 0 # 排除已存在的前景关系 fg_rels
    is_bgcand = is_cand.nonzero() # 把最后的非 0 项拿到，就是 roi 之间可能存在的背景关系

    fg_rels = fg_rels[fg_rels[:, 3] != -1]  # 使用布尔索引，移除前景关系中谓词为 -1:redundant_pred 的关系
    num_fg = min(fg_rels.size(0), int(RELS_PER_IMG * REL_FG_FRACTION * num_im)) # If too many then sample
    if num_fg < fg_rels.size(0):
        fg_rels = random_choose(fg_rels, num_fg)

    # If too many then sample，背景关系太多会进行随机采样，得到最后 bg_rels
    num_bg = min(is_bgcand.size(0) if is_bgcand.dim() > 0 else 0,
                 int(RELS_PER_IMG * num_im) - num_fg)

    if not add_bg_rels:
        num_bg = 0

    if num_bg > 0:
        bg_rels = torch.cat((
            im_inds[is_bgcand[:, 0]][:, None],
            is_bgcand,
            (is_bgcand[:, 0, None] < -10).long(),
        ), 1)

        if num_bg < is_bgcand.size(0):
            bg_rels = random_choose(bg_rels, num_bg)

        rel_labels = torch.cat((fg_rels, bg_rels), 0) # 合并前景关系与背景关系，作为最后的返回值
    else:
        rel_labels = fg_rels

    # last sort by rel. 按图像索引、第一个对象索引和第二个对象索引排序。
    _, perm = torch.sort(rel_labels[:, 0]*(gt_boxes.size(0)**2) +
                         rel_labels[:,1]*gt_boxes.size(0) + rel_labels[:,2])

    rel_labels = rel_labels[perm].contiguous() # 显存中密集排列

    return rel_labels