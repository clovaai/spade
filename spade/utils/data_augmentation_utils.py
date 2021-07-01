# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

from copy import deepcopy

import cv2
import numpy as np


def image_rotation(image, size, boxes, clip_box_coord, angle=10):
    """
    INPUT
        image : numpy array (cv2 image) or None
        size  : (width, height) of image
        boxes : coordinates -> [[x, y], [x, y], ...]
        angle : degree -> 0, 90, 180, 270, ...

    OUTPUT
        img : rotated image
        nboxes : new box coordinates -> [[x, y], [x, y], ...]
    """
    if angle == 0:
        return boxes

    (width, height) = size
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    if image is not None:
        img = np.copy(image)
        img = cv2.warpAffine(img, M, size)
    else:
        img = image

    nboxes = []
    if boxes is not None:
        for box in boxes:
            nbox = []
            for xy in box:
                nxy = M.dot(np.append(xy, 1)).tolist()

                if clip_box_coord:
                    nx, ny = nxy
                    nx = np.clip(nx, 0, width - 1)
                    ny = np.clip(ny, 0, height - 1)
                    nxy = [nx, ny]
                nbox.append(nxy)
            nboxes.append(nbox)

    return img, nboxes


def image_warping(
    image,
    size,
    boxes,
    clip_box_coord,
    n=2.0,
    amp=15.0,
    direction=0,
    normalize_amp=False,
):
    """
    INPUT
        image : numpy array (cv2 image) or None
        size  : (width, height) of image
        n     : the number of sine wave(s)
        amp   : the amplitude of sine wave(s)
        direction : sine wave direction -> 0: vertical  |  1: horizontal
    """

    cols, rows = size
    # todo: use relative amplitude

    if normalize_amp:
        # ave_size = (cols + rows)/2
        ave_size = cols
        move_func = lambda x: amp / 1000 * ave_size * np.sin(2 * np.pi * x / rows * n)
    else:
        move_func = lambda x: amp * np.sin(2 * np.pi * x / rows * n)

    img = image
    if image is not None:
        img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(rows):
            for j in range(cols):
                if direction == 1:
                    offset_x = int(round(move_func(i)))
                    offset_y = 0
                    new_j = j + offset_x
                    if new_j < cols and new_j >= 0:
                        img[i, j] = image[i, new_j % cols]
                    else:
                        img[i, j] = 0

                else:
                    offset_x = 0
                    offset_y = int(round(move_func(j)))
                    new_i = i + offset_y
                    if new_i < rows and new_i >= 0:
                        img[i, j] = image[new_i % rows, j]
                    else:
                        img[i, j] = 0

    nboxes = []
    if boxes is not None:
        for box in boxes:
            nbox = []
            assert len(box) == 4
            for i in range(4):
                x = box[i][0]
                y = box[i][1]
                if direction == 1:
                    nx = x - move_func(y)
                    ny = y
                else:
                    ny = y - move_func(x)
                    nx = x

                if clip_box_coord:
                    nx = np.clip(nx, 0, cols - 1)
                    ny = np.clip(ny, 0, rows - 1)

                nbox.append([nx, ny])
            nboxes.append(nbox)

    return img, nboxes


# img = cv2.imread('/path/to/image.png')
#
# dimensions = img.shape


def cal_merge_offset(horizontal_merge, h1, w1, h2, w2):
    if horizontal_merge:
        w = w1 + w2
        h = max(h1, h2)

        if h1 < h2:
            i_st1 = (h2 - h1) // 2
            i_ed1 = i_st1 + h1
            j_st1 = 0
            j_ed1 = j_st1 + w1

            i_st2 = 0
            i_ed2 = i_st2 + h2
            j_st2 = j_ed1
            j_ed2 = j_st2 + w2

        else:
            i_st1 = 0
            i_ed1 = i_st1 + h1
            j_st1 = 0
            j_ed1 = j_st1 + w1

            i_st2 = (h1 - h2) // 2
            i_ed2 = i_st2 + h2
            j_st2 = j_ed1
            j_ed2 = j_st2 + w2

    else:
        w = max(w1, w2)
        h = h1 + h2

        if w1 < w2:
            j_st1 = (w2 - w1) // 2
            j_ed1 = j_st1 + w1
            i_st1 = 0
            i_ed1 = i_st1 + h1

            i_st2 = i_ed1
            i_ed2 = i_st2 + h2
            j_st2 = 0
            j_ed2 = j_st2 + w2

        else:
            j_st1 = 0
            j_ed1 = j_st1 + w1
            i_st1 = 0
            i_ed1 = i_st1 + h1

            i_st2 = i_ed1
            i_ed2 = i_st2 + h2
            j_st2 = (w1 - w2) // 2
            j_ed2 = j_st2 + w2

    return h, w, i_st1, i_ed1, j_st1, j_ed1, i_st2, i_ed2, j_st2, j_ed2


def gen_merged_coord(coord, i_st, j_st):
    new_coord = np.array(coord)
    new_coord[:, :, 0] = new_coord[:, :, 0] + j_st
    new_coord[:, :, 1] = new_coord[:, :, 1] + i_st

    return new_coord.tolist()


def get_col_idx_of_rep_field_value(
    n_field, fields, field_rs, field_rel_mat, grp_rel_mat
):
    """

    Args:
        rel_mat: numpy array with shape [n_row, n_col]
    """
    # target_idx = np.nonzero(np.sum(grp_rel_mat[n_field:, :], axis=1))[0]
    target_idx = []
    for field_r in field_rs:
        id_field = fields.index(field_r)
        target_idx += np.nonzero(field_rel_mat[id_field])[0].tolist()

    return target_idx


def gen_merged_label(
    label1, label2, fields1, fields2, field_rs1, field_rs2, field_r_top1, field_r_top2
):
    label1 = np.array(label1)
    label2 = np.array(label2)

    n_depth1, n_row1, n_col1 = label1.shape
    n_depth2, n_row2, n_col2 = label2.shape

    n_field1 = len(fields1)
    n_field2 = len(fields2)

    assert fields1 == fields2
    assert field_r_top1 == field_r_top2
    assert n_row1 == n_col1 + n_field1
    assert n_row2 == n_col2 + n_field2
    assert n_depth1 == n_depth2
    assert n_depth1 == 2

    # assert n_field1 >= n_field2 #
    # for field2 in fields2:
    #     assert field2 in fields1
    # otherwise, we need more complex algorithm.

    # use ficticious root field for grouping as it is of top in hierarchy
    n_root = 2
    fields = [f"root.{i}" for i in range(n_root)] + fields1
    n_field = len(fields)
    field_rs = deepcopy(field_rs1)
    n_depth = n_depth1
    n_row = n_root + n_row1 + n_row2 - n_field2
    n_col = n_col1 + n_col2

    new_label = np.zeros([n_depth, n_row, n_col])

    # construct field grouping (lowest order

    for k in range(n_depth1):
        new_label[k, n_root : (n_root + n_row1), 0:n_col1] = label1[k, :, :]
        new_label[k, n_root:n_field, n_col1:n_col] = label2[k, :n_field2, :]
        new_label[k, (n_root + n_row1) :, n_col1:n_col] = label2[k, n_field1:, :]

    # consider 3rd relation as there is no root field of top level for two column receipt.
    grp_col_idx1 = get_col_idx_of_rep_field_value(
        n_field1, fields1, field_rs1, label1[0, :, :], label1[1, :, :]
    )
    # print(grp_col_idx1)
    new_label[0, 0, grp_col_idx1] = 1

    grp_col_idx2 = get_col_idx_of_rep_field_value(
        n_field2, fields2, field_rs2, label2[0, :, :], label2[1, :, :]
    )
    grp_col_idx2_offsetted = [n_col1 + x for x in grp_col_idx2]
    new_label[0, 1, grp_col_idx2_offsetted] = 1

    new_label = new_label.astype(int).tolist()

    return new_label, fields, field_rs


def gen_merged_label_with_new_relation(
    label1, label2, fields1, fields2, field_rs1, field_rs2, field_r_top1, field_r_top2
):
    label1 = np.array(label1)
    label2 = np.array(label2)

    n_depth1, n_row1, n_col1 = label1.shape
    n_depth2, n_row2, n_col2 = label2.shape

    n_field1 = len(fields1)
    n_field2 = len(fields2)

    assert fields1 == fields2
    assert field_r_top1 == field_r_top2
    assert n_row1 == n_col1 + n_field1
    assert n_row2 == n_col2 + n_field2
    assert n_depth1 == n_depth2
    assert n_depth1 == 2

    # assert n_field1 >= n_field2 #
    # for field2 in fields2:
    #     assert field2 in fields1
    # otherwise, we need more complex algorithm.

    # use ficticious root field for grouping as it is of top in hierarchy
    n_root = 2
    fields = [f"root.{i}" for i in range(n_root)] + fields1
    n_field = len(fields)
    field_rs = deepcopy(field_rs1)
    n_depth = n_depth1 + 1
    n_row = n_root + n_row1 + n_row2 - n_field2
    n_col = n_col1 + n_col2

    new_label = np.zeros([n_depth, n_row, n_col])

    # construct field grouping (lowest order

    for k in range(n_depth1):
        new_label[k, n_root : (n_root + n_row1), 0:n_col1] = label1[k, :, :]
        new_label[k, n_root:n_field, n_col1:n_col] = label2[k, :n_field2, :]
        new_label[k, (n_root + n_row1) :, n_col1:n_col] = label2[k, n_field1:, :]

    # consider 3rd relation as there is no root field of top level for two column receipt.
    grp_col_idx1 = get_col_idx_of_rep_field_value(
        n_field1, fields1, field_rs1, label1[0, :, :], label1[1, :, :]
    )
    # print(grp_col_idx1)
    new_label[n_depth - 1, 0, grp_col_idx1] = 1

    grp_col_idx2 = get_col_idx_of_rep_field_value(
        n_field2, fields2, field_rs2, label2[0, :, :], label2[1, :, :]
    )
    grp_col_idx2_offsetted = [n_col1 + x for x in grp_col_idx2]
    new_label[n_depth - 1, 1, grp_col_idx2_offsetted] = 1

    new_label = new_label.astype(int).tolist()

    return new_label, fields, field_rs
