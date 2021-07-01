# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import base64
import json
import os
import random as python_random
import urllib
from functools import reduce

import numpy as np
import requests
from PIL import Image, ImageFile

import spade.utils.general_utils as gu
from spade.utils.data_augmentation_utils import image_rotation, image_warping


def gen_augmented_coord(
    img, coord, img_sz, coord_aug_params, clip_box_coord, normalize_amp=False
):
    if img is None:
        width_and_height = img_sz["width"], img_sz["height"]
    else:
        width_and_height = img.shape[1], img.shape[0]

    if type(coord_aug_params[0]) == list:
        param_w1, param_w2, param_r = coord_aug_params
        n_min1, n_max1, amp_min1, amp_max1 = param_w1
        n_min2, n_max2, amp_min2, amp_max2 = param_w2
        angle_min, angle_max = param_r
    else:
        n_min, n_max, amp_min, amp_max, angle_min, angle_max = coord_aug_params
        n_min1, n_max1, amp_min1, amp_max1 = n_min, n_max, amp_min, amp_max
        n_min2, n_max2, amp_min2, amp_max2 = n_min, n_max, amp_min, amp_max

    n = python_random.uniform(n_min1, n_max1)
    n2 = python_random.uniform(n_min2, n_max2)

    amp = python_random.uniform(amp_min1, amp_max1)
    amp2 = python_random.uniform(amp_min2, amp_max2)

    angle = python_random.uniform(angle_min, angle_max)

    # random switching of amp (when min max both are positive
    r = python_random.random()
    amp = amp if r > 0.5 else -amp
    r2 = python_random.random()
    amp2 = amp2 if r2 > 0.5 else -amp2

    img_w, nboxes_w = image_warping(
        img,
        width_and_height,
        coord,
        clip_box_coord,
        n,
        amp,
        direction=0,
        normalize_amp=normalize_amp,
    )

    img_w2, nboxes_w2 = image_warping(
        img_w,
        width_and_height,
        nboxes_w,
        clip_box_coord,
        n2,
        amp2,
        direction=1,
        normalize_amp=normalize_amp,
    )

    img_r, nboxes_r = image_rotation(
        img_w2, width_and_height, nboxes_w2, clip_box_coord, angle
    )

    return img_r, nboxes_r


def gen_augmented_text_tok1(token_pool, text_tok1, token_aug_params):
    """
    Example
    text_tok1 = ['App', "##le", "Juice"]
    """
    p_del, p_subs, p_insert, p_tail_insert, n_max_insert = token_aug_params
    cum_ps = np.cumsum([p_del, p_subs, p_insert])
    new_text_tok1 = []
    for token in text_tok1:
        r_middle = python_random.random()
        if r_middle < cum_ps[0]:
            # delete
            pass
        elif r_middle < cum_ps[1]:
            # replace (substitute)
            new_token = python_random.choice(token_pool)
            new_text_tok1.append(new_token)
        elif r_middle < cum_ps[2]:
            # insert in front of token
            n_target = python_random.randint(1, n_max_insert)
            new_tokens = python_random.choices(token_pool, k=n_target)
            new_text_tok1.extend(new_tokens)
        else:
            new_text_tok1.append(token)

        r_tail = python_random.random()
        # if new_text_tok1 is empty, insert new token always
        if r_tail < p_tail_insert or len(new_text_tok1) == 0:
            # insert new tokens.
            n_target = python_random.randint(1, n_max_insert)
            new_tokens = python_random.choices(token_pool, k=n_target)
            new_text_tok1.extend(new_tokens)

    return new_text_tok1


def gen_token_pool(raw_data_type, tokenizer, normalized_raw_data):
    token_pool = gen_token_pool_from_feature(
        raw_data_type, tokenizer, normalized_raw_data
    )
    token_pool = gu.remove_duplicate_in_1d_list(token_pool)
    return token_pool


def gen_token_pool_from_feature(raw_data_type, tokenizer, normalized_raw_data):
    token_pool = []
    for data1 in normalized_raw_data:
        if raw_data_type == "type0":
            words = data1["ocr_feature"]["text"]
        elif raw_data_type == "type1":
            words = data1["text"]
        else:
            raise NotImplementedError
        tokss = [tokenizer.tokenize(word) for word in words]
        toks = reduce(lambda x, y: x + y, tokss, [])
        token_pool += toks

    return token_pool


def tokenizing_func(tokenizer, text1):
    return tokenizer.tokenize(text1)


def _get_api_meta_info(url, postprocessor, detector, recognizer):
    return {
        "url": url,
        "postprocessor": postprocessor,
        "detector": detector,
        "recognizer": recognizer,
    }


def get_api_meta_info(url, api_meta_info=None):
    # read from api
    r = requests.get("/".join([url, "v1", "models"]))
    if r.status_code == 200:
        api_meta_info = r.json()
        return _get_api_meta_info(
            url,
            api_meta_info["postprocessor"],
            api_meta_info["detector"],
            api_meta_info["recognizer"],
        )

    # when read from release file, ocr_api_info is given
    if api_meta_info:
        return _get_api_meta_info(
            url,
            api_meta_info["postprocessor"],
            api_meta_info["detector"],
            api_meta_info["recognizer"],
        )
    else:  # for releases <= v1.5.6 compatibility
        ocr_api_info_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ocr_api_info.json")
        )
        if os.path.exists(ocr_api_info_file):
            with open(ocr_api_info_file) as f:
                api_meta_info = json.load(f)
                return _get_api_meta_info(
                    url,
                    api_meta_info["postprocessor"],
                    api_meta_info["detector"],
                    api_meta_info["recognizer"],
                )

    # neither release nor api-accessible
    print("warning: cannot retrieve api meta info from url %s" % url)
    return {
        url,
        "unknown",
        "unknown",
        "unknown",
    }


def quad2list2(quad):
    """
    convert to list of list
    """
    return [
        [quad[0]["x"], quad[0]["y"]],
        [quad[1]["x"], quad[1]["y"]],
        [quad[2]["x"], quad[2]["y"]],
        [quad[3]["x"], quad[3]["y"]],
    ]


def get_label_and_feature(raw_data1, task, fields, field_rs):
    if task == "receipt_v1":
        label, feature = get_adj_mat_receipt_v1(fields, field_rs, raw_data1)
    elif task == "funsd":
        label, feature = get_adj_mat_funsd(fields, field_rs, raw_data1)
    else:
        raise NotImplementedError

    return label, feature


def get_label_and_feature_infer_mode(task, raw_data1):
    label = None
    if task in ["receipt_v1"]:
        feature = [
            (x["text"], x["boundingBox"], x["isVertical"], x["confidence"])
            for x in raw_data1["words"]
        ]
        confidence = [x[3] for x in feature]
        img_sz = raw_data1["meta"]["img_size"][0]
        data_id = raw_data1["meta"]["image_id"]
    else:
        raise NotImplementedError

    return label, feature, confidence, img_sz, data_id


def get_meta_feature(task, raw_data1, feature):
    if task in ["receipt_v1", "funsd"]:
        img_sz = raw_data1["meta"]["image_size"]
        confidence = [1] * len(feature)
        data_id = raw_data1["meta"]["image_id"]
    else:
        raise NotImplementedError

    return img_sz, confidence, data_id


def quad2list2_receipt_v1(quad):
    """
    convert to list of list
    """

    return [
        [quad["x1"], quad["y1"]],
        [quad["x2"], quad["y2"]],
        [quad["x3"], quad["y3"]],
        [quad["x4"], quad["y4"]],
    ]


def coord_to_quad_receipt_v1(coord):
    quad = {
        "x1": coord[0][0],
        "y1": coord[0][1],
        "x2": coord[1][0],
        "y2": coord[1][1],
        "x3": coord[2][0],
        "y3": coord[2][1],
        "x4": coord[3][0],
        "y4": coord[3][1],
    }

    return quad


def gen_sorted_features_grouped_by_field(
    fields, all_unsorted_features_grouped_by_field
):
    # file open and get text
    sorted_features_grouped_by_field = []  # ordered
    # [
    #   [
    #       (label, None, None, gid, flag_field_of_interest, None, None, None),
    #           (text1, coord1, vertical1, gid, flag_field_of_interest, is_key, row_id, x1),
    #           (text2, corrd2, vertical2, None), ....
    #       ],
    #       [
    #       (label', ...)
    #       (text1', ...)
    #       ], ...
    # ]
    # new_pnt_list_batch1 = sorted(candidates1, key=lambda list1: list1[-1], reverse=True)

    for (
        all_unsorted_features_grouped_by_field1
    ) in all_unsorted_features_grouped_by_field:
        field_of_feature = all_unsorted_features_grouped_by_field1["category"]
        gid = all_unsorted_features_grouped_by_field1["group_id"]
        segments = all_unsorted_features_grouped_by_field1["words"]

        flag_field_of_interest = field_of_feature in fields

        label_aug = [
            (
                f"[{field_of_feature}]",
                None,
                None,
                gid,
                flag_field_of_interest,
                None,
                None,
                None,
            )
        ]

        feature_unsorted = []
        for segment in segments:
            text = segment["text"]
            row_id = segment["row_id"]
            quad = segment["quad"]
            coord = quad2list2_receipt_v1(quad)
            x1 = quad["x1"]
            is_key = segment["is_key"]

            if len(text.strip()) > 0:
                feature_unsorted.append(
                    (
                        text,
                        coord,
                        0,
                        gid,
                        flag_field_of_interest,
                        is_key,
                        row_id,
                        x1,
                    )  # always horizontal.
                )

        # 1. row sort
        feature_row_sorted = sorted(feature_unsorted, key=lambda x: x[-2])
        # 2. x-sort while conserving row-order
        rid = [x[-2] for x in feature_row_sorted]
        rid_b = np.diff(rid)  # boundary
        assert sum(rid_b < 0) == 0
        idx_bs = (
            np.nonzero(rid_b)[0] + 1
        )  # boundary idx. idx in a sence of "pythonic" (it is of between elements)
        # add final boundary (end-point)
        idx_bs = np.append(idx_bs, len(feature_row_sorted))

        feature_sorted = []  # now x1-based sort
        st = 0
        for idx_b in idx_bs:
            ed = idx_b
            feature_sorted_part = sorted(feature_row_sorted[st:ed], key=lambda x: x[-1])
            feature_sorted += feature_sorted_part
            st = ed

        # check
        # rid = [x[-2] for x in feature_sorted]
        # rid_b = np.diff(rid)  # boundary
        # assert sum(rid_b < 0) == 0
        #
        # xid = [x[-1] for x in feature_sorted]
        # np.diff(xid)

        # feature = [x[:3] for x in feature_sorted]
        # feature = [x[:4] for x in feature_sorted]

        sorted_features_grouped_by_field.append(label_aug + feature_sorted)

    return sorted_features_grouped_by_field


def gen_adj_mat_f_and_cols(
    sorted_features_grouped_by_field,
    row_offset,
    n_row,
    n_col,
    braced_fields,
    braced_field_rs,
):
    id_offset_field_to_first_box = 1000000
    adj_mat_f = np.zeros([n_row, n_col], dtype=int)  # field
    cols = []
    col_ids_rp = []  # representers
    col_ids_non_key_header = []  # first segment of each field-group.
    col_gids = []
    i_col = -1
    for sorted_features_grouped_by_field1 in sorted_features_grouped_by_field:
        label_aug = sorted_features_grouped_by_field1[0]
        feature = sorted_features_grouped_by_field1[1:]
        field_of_feature = label_aug[0]

        non_key_header_found = False
        if field_of_feature in braced_fields:
            most_recent_non_key_col_id = (
                braced_fields.index(field_of_feature) + id_offset_field_to_first_box
            )
            # + 1000000 to avoid the case where col_id == field_id which casues duplicate field to first boxes edges.

        for i_feat, feature1 in enumerate(feature):

            is_key = feature1[5]
            assert is_key == 0 or is_key == 1

            i_col += 1
            i_row = i_col + row_offset
            cols.append(feature1[:3])
            col_gids.append(feature1[3])

            # collect representer ids
            if not non_key_header_found:
                if is_key:
                    pass
                else:
                    non_key_header_found = True
                    if field_of_feature in braced_field_rs:
                        # print(field_of_feature, i_col, feature1)
                        col_ids_rp.append(i_col)
                    if field_of_feature in braced_fields:
                        col_ids_non_key_header.append(i_col)

            if field_of_feature in braced_fields:
                if is_key:
                    pass
                else:
                    if most_recent_non_key_col_id == (
                        braced_fields.index(field_of_feature)
                        + id_offset_field_to_first_box
                    ):
                        adj_mat_f[
                            most_recent_non_key_col_id - id_offset_field_to_first_box,
                            i_col,
                        ] = 1
                    else:
                        adj_mat_f[row_offset + most_recent_non_key_col_id, i_col] = 1
                    most_recent_non_key_col_id = i_col

    assert n_col == i_col + 1

    return adj_mat_f, cols, col_ids_rp, col_ids_non_key_header, col_gids


def gen_adj_mat_g(
    row_offset, n_row, n_col, col_ids_rp, col_ids_non_key_header, col_gids
):
    adj_mat_g = np.zeros([n_row, n_col], dtype=int)  # group
    for col_id_rp in col_ids_rp:
        target_gid = col_gids[col_id_rp]
        target_col_ids = [
            col_id
            for col_id, gid in enumerate(col_gids)
            if (gid == target_gid)
            and (col_id != col_id_rp)
            and (col_id in col_ids_non_key_header)
        ]

        mu = row_offset + col_id_rp
        for nu in target_col_ids:
            adj_mat_g[mu, nu] = 1

    return adj_mat_g


def get_field_collecting_idxs(gid_and_fields, gid_and_fields_unique):
    field_collecting_idxs = []
    for gid_and_field_ref in gid_and_fields_unique:
        collecting_idx_bool = [x == gid_and_field_ref for x in gid_and_fields]
        collecting_idx = np.nonzero(collecting_idx_bool)[0].tolist()
        field_collecting_idxs.append(collecting_idx)

    return field_collecting_idxs


def recollect_fields(all_unsorted_features_grouped_by_field):
    # recollect unnecessarily separted fields in GT
    gid_and_fields = [
        (x["group_id"], x["category"]) for x in all_unsorted_features_grouped_by_field
    ]
    gid_and_fields_unique = gu.remove_duplicate_in_1d_list(gid_and_fields)
    field_collecting_idxs = get_field_collecting_idxs(
        gid_and_fields, gid_and_fields_unique
    )

    # generation
    all_unsorted_features_grouped_by_recollected_field = []
    for field_collecting_idx, gid_and_fields_unique1 in zip(
        field_collecting_idxs, gid_and_fields_unique
    ):
        # insert
        gid, field_of_feature = gid_and_fields_unique1
        words = []
        for field_collecting_idx1 in field_collecting_idx:
            words += all_unsorted_features_grouped_by_field[field_collecting_idx1][
                "words"
            ]
        all_unsorted_features_grouped_by_recollected_field.append(
            {"group_id": gid, "category": field_of_feature, "words": words}
        )

    # assertion
    if len(gid_and_fields) == len(gid_and_fields_unique):
        assert gid_and_fields == gid_and_fields_unique
        assert (
            all_unsorted_features_grouped_by_field
            == all_unsorted_features_grouped_by_recollected_field
        )
    else:
        # print('hi')
        pass

    return all_unsorted_features_grouped_by_recollected_field


def get_adj_mat_receipt_v1(fields, field_rs, raw_data1, verbose=False):
    """
    raw_data1.keys(): ['dontcare', 'valid_line', 'meta', 'roi', 'repeating_symbol']

    raw_data1['valid_line'][0].keys(): ['words', 'category', 'group_id']

        "group_id": null if "O" otherwise, it indicates literally "group" info. integer.
        "category": null if "O" otherwise, it indicates sub-group category. e.g.) "menu.discountprice"

    raw_data1['valid_line'][0]['words'][0].keys() : ['quad', 'is_key', 'row_id', 'text']
        is_key: 0 or 1
        row_id:  e.g. 600222 (should be offset)
        "text": e.g. "soda"
    ETC:
        - No vertical info. presented. "Horizontal" assumed.
        - use "row_id" and "x" for relation tagging.
        - use "group_id" for 1st groupping.
            - second grouping should be based on "valid_line".
    """

    # 1. Generate cols and rows (sorted and grouped_by_field)
    all_unsorted_features_grouped_by_field = raw_data1["valid_line"]
    # cols = [ (text1, coord1, vertical1, gid, flag_field_of_interest, is_key, row_id, x1), (), ... ]

    # 2. Combine fields having same gid
    all_unsorted_features_grouped_by_recollected_field = recollect_fields(
        all_unsorted_features_grouped_by_field
    )

    sorted_features_grouped_by_field = gen_sorted_features_grouped_by_field(
        fields, all_unsorted_features_grouped_by_recollected_field
    )

    # 3. adj_mat making
    row_offset = len(fields)
    n_col = sum(
        [
            len(label_and_feature[1:])
            for label_and_feature in sorted_features_grouped_by_field
        ]
    )
    n_row = n_col + len(fields)
    braced_fields = [f"[{f1}]" for f1 in fields]
    braced_field_rs = [f"[{f1}]" for f1 in field_rs]

    (
        adj_mat_f,
        cols,
        col_ids_rp,
        col_ids_non_key_header,
        col_gids,
    ) = gen_adj_mat_f_and_cols(
        sorted_features_grouped_by_field,
        row_offset,
        n_row,
        n_col,
        braced_fields,
        braced_field_rs,
    )

    adj_mat_g = gen_adj_mat_g(
        row_offset, n_row, n_col, col_ids_rp, col_ids_non_key_header, col_gids
    )

    return [adj_mat_f.tolist(), adj_mat_g.tolist()], cols


def gen_box_id_and_contents(rows):
    gen_box_id_and_contents = {}
    for row in rows:
        row_id = row["row_id"]
        boxes = row["boxes"]
        for box in boxes:
            box_id = box["box_id"]
            assert box_id not in gen_box_id_and_contents

            contents = {
                "row_id": row_id,
                "quad": box["quad"],
                "text_full_box": box["text"],
            }

            gen_box_id_and_contents[box_id] = contents
    return gen_box_id_and_contents


def gen_sub_box(quad_full_box, st_idx, ed_idx, l_text_full_box, vertical1):
    coord_full_box = np.array(quad2list2_receipt_v1(quad_full_box))

    direction_vec = get_direction_vec(coord_full_box, vertical1)
    if l_text_full_box > 0:
        dvec_char = direction_vec / l_text_full_box
    else:
        # empty string due to recognition failure I guess.
        # set lenght as 1
        dvec_char = direction_vec

    coord_sub_box = get_coord1_first_char(
        coord_full_box, dvec_char, vertical1, st_idx, ed_idx
    )
    quad_sub_box = coord_to_quad_receipt_v1(coord_sub_box)

    return quad_sub_box, coord_sub_box


def insert_coord_into_sub_groups(sub_groups, box_id_and_contents):
    all_unsorted_features_grouped_by_field = []
    box_id_and_used_text_spans = {k: [] for k in box_id_and_contents}

    for sub_group in sub_groups:
        all_unsorted_feature_grouped_by_field = {
            "group_id": sub_group["group_id"],
            "category": sub_group["category"],
            "words": [],
        }

        for word in sub_group["words"]:
            box_id = word["box_id"]
            st_idx = word["start_index"]
            ed_idx = word["end_index"]

            text = word["text"]

            quad_full_box = box_id_and_contents[box_id]["quad"]
            text_full_box = box_id_and_contents[box_id]["text_full_box"]
            l_text_full_box = len(text_full_box)
            assert text_full_box[st_idx:ed_idx] == text
            ed_idx = (
                ed_idx if ed_idx <= l_text_full_box else l_text_full_box
            )  # v1.4.1 bug (ed_idx can be larger than len (when there is no_split)

            quad_sub_box, _ = gen_sub_box(
                quad_full_box, st_idx, ed_idx, l_text_full_box, vertical1=False
            )

            new_word = {
                "quad": quad_sub_box,
                "is_key": word["is_key"],
                "row_id": box_id_and_contents[box_id]["row_id"],
                "text": text,
            }

            all_unsorted_feature_grouped_by_field["words"].append(new_word)
            box_id_and_used_text_spans[box_id].append((st_idx, ed_idx, l_text_full_box))

        all_unsorted_features_grouped_by_field.append(
            all_unsorted_feature_grouped_by_field
        )

    return all_unsorted_features_grouped_by_field, box_id_and_used_text_spans


def gen_unused_text_span_from_unused_char_ids(char_unused):
    unused_text_span = []
    span1 = [0, None] if char_unused[0] else None

    for i in range(len(char_unused) - 1):
        char_flag_before = char_unused[i]
        char_flag = char_unused[i + 1]
        i_char = i + 1

        if char_flag:

            if span1 is None:
                st = i_char
                span1 = [st, None]

                if i_char == len(char_unused) - 1:
                    span1 = [st, st + 1]
                    unused_text_span.append(span1)
                    span1 = None
            else:
                # update ed id
                ed = i_char + 1
                span1[1] = ed
                if i_char == len(char_unused) - 1:
                    unused_text_span.append(span1)
                    span1 = None
        else:
            if char_flag_before:
                ed = i_char  # no + 1 as it is ended before.
                span1[1] = ed
                unused_text_span.append(span1)
                span1 = None

            else:
                pass
    return unused_text_span


def gen_flag_char_unused(used_text_span):
    _, _, l_full_text = used_text_span[0]
    char_unused = np.array([True] * l_full_text)
    for used_text_span1 in used_text_span:
        st, ed, _l = used_text_span1
        assert _l == l_full_text
        char_unused[st:ed] = False
    return char_unused.tolist()


def get_unused_text_span(used_text_span):
    """
    [(st, ed, l_full_text), (st, ed, l_full_text), ...]
    """
    char_unused = gen_flag_char_unused(used_text_span)
    unused_text_span = gen_unused_text_span_from_unused_char_ids(char_unused)

    return unused_text_span


def gen_o_cols(box_id_and_contents, box_id_and_used_text_spans):
    is_vertical = False
    o_cols = []
    for box_id, box_contents in box_id_and_contents.items():
        used_text_span = box_id_and_used_text_spans[box_id]
        if used_text_span:
            if (len(used_text_span) == 1) and (used_text_span[0][2] == 0):
                unused_text_span = []
                # blank string
            else:
                unused_text_span = get_unused_text_span(used_text_span)
        else:
            st = 0
            ed = len(box_contents["text_full_box"])
            unused_text_span = [(st, ed)]

        for unused_text_span1 in unused_text_span:
            st, ed = unused_text_span1
            quad_full_box = box_id_and_contents[box_id]["quad"]
            text_full_box = box_id_and_contents[box_id]["text_full_box"]
            l_full_text = len(text_full_box)
            _, coord_sub_box = gen_sub_box(
                quad_full_box, st, ed, l_full_text, vertical1=is_vertical
            )

            if len(text_full_box[st:ed].strip()) > 0:
                o_col = (
                    text_full_box[st:ed],
                    [x.tolist() for x in coord_sub_box],
                    int(is_vertical),
                )
                o_cols.append(o_col)

    return o_cols


def gen_and_insert_coord(sub_groups, rows):
    # search corresponding box from ocr-output using box_id
    # split box and generate new quad based on the start_index and end_index

    box_id_and_contents = gen_box_id_and_contents(rows)
    (
        all_unsorted_features_grouped_by_field,
        box_id_and_used_text_spans,
    ) = insert_coord_into_sub_groups(sub_groups, box_id_and_contents)
    o_cols = gen_o_cols(box_id_and_contents, box_id_and_used_text_spans)

    return all_unsorted_features_grouped_by_field, o_cols


def convert_v2_format_to_v1_format(raw_data1):
    sub_groups = raw_data1["sub_groups"]
    rows = raw_data1["rows"]
    assert len(raw_data1["dontcare"]) == 0

    all_unsorted_features_grouped_by_field, o_cols = gen_and_insert_coord(
        sub_groups, rows
    )
    raw_data1["valid_line"] = all_unsorted_features_grouped_by_field
    return raw_data1, o_cols


def concat_o_cols(adj_mat_fg, cols, o_cols):
    nr = len(adj_mat_fg[0])
    nc = len(adj_mat_fg[0][0])
    n_o_cols = len(o_cols)
    nnr = nr + n_o_cols
    nnc = nc + n_o_cols

    new_adj_mat_fg = []
    for adj_mat1 in adj_mat_fg:
        new_adj_mat1 = []
        for row in adj_mat1:
            new_adj_mat1.append(row + [0] * n_o_cols)
        for _ in range(n_o_cols):
            new_adj_mat1.append([0] * nnc)

        new_adj_mat_fg.append(new_adj_mat1)

    new_cols = cols + o_cols
    # assert nnr == len(new_adj_mat_fg[0])
    # assert nnc == len(new_adj_mat_fg[0][0])
    return new_adj_mat_fg, new_cols


def funsd_box_to_coord(box):
    x1, y1, x3, y3 = box
    x2, y2 = x3, y1
    x4, y4 = x1, y3

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


def check_assumption_on_linking(linking, id_target, id_to_label_ori):
    target_is_head = id_target == linking[0]

    h_label_ori = id_to_label_ori[linking[0]]
    d_label_ori = id_to_label_ori[linking[1]]
    raise NotImplementedError

    return


def get_some_stat_from_form_checking_data_structure(form):
    l_box_at_each_form = [len(x["words"]) for x in form]
    l_word = sum(l_box_at_each_form)

    id_to_label_ori = {}
    form_id_to_first_col_id = {}
    i_col = -1
    for i_form, form1 in enumerate(form):
        id = form1["id"]
        # assert i_form == id  # assume id is well-sorted.
        label_ori = form1["label"]

        id_to_label_ori[id] = label_ori

        for i_word1, word1 in enumerate(form1["words"]):
            i_col += 1
            if i_word1 == 0:
                x1_before = word1["box"][0]
                y1_before = word1["box"][1]
                form_id_to_first_col_id[id] = i_col
            else:
                x1_now = word1["box"][0]
                y1_now = word1["box"][1]
                # if x1_now < x1_before:
                #     assert y1_now > y1_before

    return l_word, id_to_label_ori, form_id_to_first_col_id


def scitsr_offset_coord(coord):
    min_x = min([coord1[0][0] for coord1 in coord])
    min_y = min([coord1[0][1] for coord1 in coord])

    return [
        [
            [coord1[0][0] - min_x, coord1[0][1] - min_y],
            [coord1[1][0] - min_x, coord1[1][1] - min_y],
            [coord1[2][0] - min_x, coord1[2][1] - min_y],
            [coord1[3][0] - min_x, coord1[3][1] - min_y],
        ]
        for coord1 in coord
    ]


def replace_empty_to_symbol(fname, text_segment):
    # replace empty string into "symbol" which is likely due to OCR error for math symbol
    non_empty_flag = bool(text_segment.strip())
    if not non_empty_flag:
        print(f'{fname}: Emtpy text is replaced into "symbol" string')
    return text_segment if non_empty_flag else "symbol"
    pass


def get_adj_mat_funsd(fields, field_rs, raw_data1):
    """ """

    # cols = (text, coord, is_vertical (0 or 1))

    map_label_ori_to_field = {
        "answer": "qa.answer",
        "question": "qa.question",
        "header": "header.header",
        "other": "other.other",
    }

    form = raw_data1["form"]

    (
        l_word,
        id_to_label_ori,
        form_id_to_first_col_id,
    ) = get_some_stat_from_form_checking_data_structure(form)

    n_field = len(fields)
    row_offset = n_field

    cols = []
    adj_mat_fg = np.zeros([2, n_field + l_word, l_word], dtype=np.int)
    i_col = -1
    for i_form, form1 in enumerate(form):
        word = form1["words"]
        linking = form1["linking"]  # [ [], [], ...]
        label_ori_target = form1["label"]
        field_target = map_label_ori_to_field[label_ori_target]

        fid_target = fields.index(field_target)
        id_target = form1["id"]

        # check_assumption_on_linking(linking, id_target, id_to_label_ori)

        # rel-g construction
        for linking1 in linking:
            hid, tid = linking1
            hcid = form_id_to_first_col_id[hid]
            tcid = form_id_to_first_col_id[tid]
            adj_mat_fg[1, row_offset + hcid, tcid] = 1

        # rel-s construction
        # assume words are sorted along x coordinate
        for i_word1, word1 in enumerate(word):
            i_col += 1
            is_vertical = 0
            coord1 = funsd_box_to_coord(word1["box"])

            if i_word1 == 0:
                adj_mat_fg[0, fid_target, i_col] = 1
            else:
                adj_mat_fg[0, row_offset + i_col - 1, i_col] = 1

            l_word1 = len(word1["text"])
            text1 = word1["text"] if l_word1 > 0 else "[UNK]"

            col = (text1, coord1, is_vertical, id_target, adj_mat_fg)
            cols.append(col)

    return adj_mat_fg, cols


def get_direction_vec(coord1, vertical1):
    c1, c2, c3, c4 = coord1
    # x1, y1 = c1
    # x2, y2 = c2
    # x3, y3 = c3
    # x4, y4 = c4

    if vertical1:
        direction_vec = (c3 + c4) / 2 - (c1 + c2) / 2
    else:
        direction_vec = (c2 + c3) / 2 - (c1 + c4) / 2

    return direction_vec


def augment_vertical(vertical1, l_tok1):
    return [vertical1] * l_tok1


def augment_char_size(csz, l_tok1):
    return [csz] * l_tok1


def get_coord1_first_char(coord1, dvec, vertical1, n_char11_offset, n_char11):
    c1, c2, c3, c4 = coord1

    if vertical1:
        new_c1 = c1 + dvec * n_char11_offset
        new_c2 = c2 + dvec * n_char11_offset
        new_c3 = c2 + dvec * n_char11
        new_c4 = c1 + dvec * n_char11
    else:
        new_c1 = c1 + dvec * n_char11_offset
        new_c4 = c4 + dvec * n_char11_offset
        new_c2 = c1 + dvec * n_char11
        new_c3 = c4 + dvec * n_char11
        # new_c3 = c1 + dvec * n_char11

    return [new_c1, new_c2, new_c3, new_c4]


def augment_coord(coord1, vertical1, l_tok1, method, text_tok1):
    """

    Args:
        coord1: numpy ndarray
        vertical1: bool
        l_tok1: numeric
        bag_of_words: bool

    Returns:

    """
    direction_vec = get_direction_vec(coord1, vertical1)
    direction_vecs = [direction_vec] * l_tok1

    if method == "bag_of_words":
        coord_tok1 = [coord1] * l_tok1
    elif method == "equal_division":
        """each token as if has single char width"""
        coord_tok1 = []
        dvec_tok = direction_vec / l_tok1
        coord1_fc = get_coord1_first_char(
            coord1, dvec_tok, vertical1, n_char11_offset=0, n_char11=1
        )

        for i in range(l_tok1):
            tok_pos = coord1_fc + i * dvec_tok
            coord_tok1.append(tok_pos.tolist())

    elif method == "char_lv_equal_division":
        coord_tok1 = []
        text_tok1_no_sharp = [xx.replace("#", "") for xx in text_tok1]
        n_char_text_tok1_no_sharp = [len(xx) for xx in text_tok1_no_sharp]

        # original #### token cause the problem, thus add single length for it
        for i_nchar, n_char11 in enumerate(n_char_text_tok1_no_sharp):
            if n_char11 == 0:
                n_char_text_tok1_no_sharp[i_nchar] = 1

        n_char_text_tok1_no_sharp_cumsum = np.cumsum(n_char_text_tok1_no_sharp).tolist()
        l_char1 = n_char_text_tok1_no_sharp_cumsum[-1]
        dvec = direction_vec / l_char1
        n_char11_before = 0
        for i, n_char11 in enumerate(n_char_text_tok1_no_sharp_cumsum):
            tok_pos = np.array(
                get_coord1_first_char(
                    coord1,
                    dvec,
                    vertical1,
                    n_char11_offset=n_char11_before,
                    n_char11=n_char11,
                )
            )
            n_char11_before = n_char11
            coord_tok1.append(tok_pos.tolist())
    else:
        raise NotImplementedError

    return coord_tok1, direction_vecs


def get_char_size1(coord1, v):
    """
    if not vetical
    c1          c2
    c4          c3


    if vetical
    c1  c2



    c4  c3
    """
    c1, c2, c3, c4 = coord1
    if v:
        csz = (np.linalg.norm(c1 - c2) + np.linalg.norm(c3 - c4)) / 2
    else:
        csz = (np.linalg.norm(c1 - c4) + np.linalg.norm(c2 - c3)) / 2

    return csz


def remove_target(list_in, l_str):
    tf_arr = np.array(l_str) != 0
    list_out = np.array(list_in)[tf_arr].tolist()
    return list_out


def remove_blank_box(text, coord, vertical):
    l_str = [len(x.strip()) for x in text]

    text = remove_target(text, l_str)
    coord = remove_target(coord, l_str)
    vertical = remove_target(vertical, l_str)

    return text, coord, vertical


def update_label_sub(l_tok1, rel_idx, r_pnt, c_pnt, label_sub1, type):
    def _insert_zero_row(arr, idx):
        n_row, n_col = arr.shape
        return np.insert(arr, obj=idx, values=[0] * n_col, axis=0)

    def _insert_zero_col(arr, idx):
        n_row, n_col = arr.shape
        return np.insert(arr, obj=idx, values=[0] * n_row, axis=1)

    # laberl_sub. List of adj_mat
    r_pnt_header = r_pnt
    c_pnt_header = c_pnt

    # label
    # remove arrow to the next boxes before augment label.
    next_box_col_idxs = np.where(label_sub1[r_pnt_header] == rel_idx)[0]
    if next_box_col_idxs.size > 0:
        # assert next_box_col_idxs.size == 1
        for next_box_col_idx in next_box_col_idxs:
            label_sub1[r_pnt_header, next_box_col_idx] = 0
            # make it zero as connection info needs to be moved to the final token
    else:
        next_box_col_idx = None

    # insert zero column and row.
    for i in range(l_tok1 - 1):
        label_sub1 = _insert_zero_row(label_sub1, r_pnt_header + 1)
        label_sub1 = _insert_zero_col(label_sub1, c_pnt_header + 1)

    # move pnt for generated tokens.
    for i in range(l_tok1 - 1):
        # move pnt to in front of next token
        r_pnt += 1
        c_pnt += 1
        # insert 1
        if type == "f":  # field group
            label_sub1[r_pnt - 1, c_pnt] = rel_idx
        elif type == "g":
            pass
        elif type == "root":
            pass
        else:
            raise NotImplementedError

    # re inserted previously removed arrows to the next boxes.
    if next_box_col_idxs is not None:
        # modify next_box_col_idx
        for next_box_col_idx in next_box_col_idxs:
            if next_box_col_idx <= c_pnt_header:
                j = next_box_col_idx
            else:
                j = next_box_col_idx + (l_tok1 - 1)

            if type == "f":  # field group
                label_sub1[r_pnt, j] = rel_idx
            elif type == "g":
                label_sub1[r_pnt_header, j] = rel_idx
            elif type == "root":
                pass
            else:
                raise NotImplementedError

    # r_pnt_header = r_pnt
    # c_pnt_header = c_pnt
    #
    # # label
    # next_box_col_idxs = np.where(label_sub[r_pnt_header] == rel_idx)[0]
    # if next_box_col_idxs.size > 0:
    #     assert next_box_col_idxs.size == 1
    #     next_box_col_idx = next_box_col_idxs[0]
    #     label_sub[r_pnt_header, next_box_col_idx] = 0
    #     # make it zero as connection info needs to be moved to the final token
    # else:
    #     next_box_col_idx = None
    #
    # # insert zero column and row.
    #
    # for i in range(l_tok1 - 1):
    #     label_sub = _insert_zero_row(label_sub, r_pnt_header + 1)
    #     label_sub = _insert_zero_col(label_sub, c_pnt_header + 1)
    # header_tok = np.insert(header_tok, obj=c_pnt_header + 1, values=0, axis=0)
    #
    # # move pnt for generated tokens.
    # for i in range(l_tok1 - 1):
    #     # move pnt to in front of next token
    #     r_pnt += 1
    #     c_pnt += 1
    #     # insert 1
    #     label_sub[r_pnt - 1, c_pnt] = rel_idx
    #
    # if next_box_col_idx is not None:
    #     label_sub[r_pnt, next_box_col_idx + (l_tok1 - 1)] = rel_idx

    # label_tok = [label_sub]
    return label_sub1, r_pnt, c_pnt


def char_height_normalization(n_char_unit, char_height):
    unit_len = np.min([x for x in char_height if x > 0])
    lb = 0
    ub = n_char_unit
    new_arr = np.clip(char_height / unit_len, lb, ub)
    return new_arr.astype(np.int)


def dist_normalization(
    dist_norm, n_dist_unit, arr, img_sz, char_height, all_positive=False
):
    h = img_sz["height"]
    w = img_sz["width"]

    if dist_norm == "char_height":
        unit_len = np.median(char_height)
    elif dist_norm == "img_diagonal":
        unit_len = np.sqrt(w ** 2 + h ** 2) / n_dist_unit
    else:
        raise NotImplementedError

    lb = 0 if all_positive else -n_dist_unit
    ub = n_dist_unit
    new_arr = np.clip(arr / unit_len, lb, ub)
    return new_arr.astype(np.int)


def angle_normalization(n_angle_unit, arr):
    new_arr = np.clip(arr / (2 * np.pi) * n_angle_unit, 0, n_angle_unit)
    return new_arr.astype(np.int)
