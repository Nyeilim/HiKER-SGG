######
# Draws rectangles
######

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def draw_union_boxes(bbox_pairs, pooling_size, padding=0):
    """
    Draws union boxes for the image.
    :param box_pairs: [num_pairs, 8]
    :param fmap_size: Size of the original feature map
    :param stride: ratio between fmap size and original img (<1)
    :param pooling_size: resize everything to this size
    :return: [num_pairs, 2, pooling_size, pooling_size] arr
    """
    assert padding == 0, "Padding>0 not supported yet"
    return draw_union_boxes_c(bbox_pairs, pooling_size)

cdef DTYPE_t minmax(DTYPE_t x):
    return min(max(x, 0), 1)

cdef np.ndarray[DTYPE_t, ndim=4] draw_union_boxes_c(
        np.ndarray[DTYPE_t, ndim=2] box_pairs, unsigned int pooling_size):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float. everything has arbitrary ratios
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = box_pairs.shape[0]

    cdef np.ndarray[DTYPE_t, ndim = 4] uboxes = np.zeros(
        (N, 2, pooling_size, pooling_size), dtype=DTYPE)
    cdef DTYPE_t x1_union, y1_union, x2_union, y2_union, w, h, x1_box, y1_box, x2_box, y2_box, y_contrib, x_contrib
    cdef unsigned int n, i, j, k

    for n in range(N):
        x1_union = min(box_pairs[n, 0], box_pairs[n, 4]) # 计算并集框的左上角
        y1_union = min(box_pairs[n, 1], box_pairs[n, 5])
        x2_union = max(box_pairs[n, 2], box_pairs[n, 6]) # 计算并集框的右下角
        y2_union = max(box_pairs[n, 3], box_pairs[n, 7])
        # 别被他的 Pooling 吓到了，其实就是放缩
        w = x2_union - x1_union # 并集框的宽，随后将被缩放到 27
        h = y2_union - y1_union # 并集框的高，随后将被缩放到 27
       
        for i in range(2): # 这个迭代 0 1 就是分别代表 subject,object 放缩后的框
            # Now everything is in the range [0, pooling_size]. subject,object 框小于并集框，将其坐标等比缩放到 27x27 的范围
            x1_box = (box_pairs[n, 0+4*i] - x1_union)*pooling_size / w # 对左上角的处理
            y1_box = (box_pairs[n, 1+4*i] - y1_union)*pooling_size / h
            x2_box = (box_pairs[n, 2+4*i] - x1_union)*pooling_size / w # 对右下角的处理
            y2_box = (box_pairs[n, 3+4*i] - y1_union)*pooling_size / h
            # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x1_box, y1_box, x2_box, y2_box))
            for j in range(pooling_size):
                y_contrib = minmax(j+1-y1_box)*minmax(y2_box-j)
                for k in range(pooling_size):
                    x_contrib = minmax(k+1-x1_box)*minmax(x2_box-k)                
                    # print("j {} yc {} k {} xc {}".format(j, y_contrib, k, x_contrib))
                    uboxes[n,i,j,k] = x_contrib*y_contrib
    return uboxes # 最后的返回值 shape(num_all_rels,2,27,27)，27x27 可以理解成原 boxes 映射后在其上的覆盖程度，比如你原 boxes 在 union boxes 中偏右，那你这个 27x27 的矩阵左边值就会偏低
