# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

from copy import deepcopy

import numpy as np
import torch

import spade.utils.general_utils as gu
from spade.postprocess.eval import (get_init_stats_receipt,
                                    get_statistics_receipt, summary_receipt)

get_statistics_receipt_v1 = get_statistics_receipt
get_init_stats_receipt_v1 = get_init_stats_receipt


summary_receipt_v1 = summary_receipt
summary_funsd = summary_receipt


def cal_parsing_score(
    mode,
    parses,
    pr_parses,
    task,
    reformat,
    refine_parse=False,
    return_refined_parses=False,
    allow_small_edit_distance=False,
    task_lan="",
    unwanted_fields=[],
):
    get_init_stats_task = eval(f"get_init_stats_{task}")
    summary_task = eval(f"summary_{task}")

    if unwanted_fields:
        parses = trim_parse_for_scoring(
            parses, unwanted_fields, is_shallow=not reformat[0]
        )
        pr_parses = trim_parse_for_scoring(
            pr_parses, unwanted_fields, is_shallow=not reformat[1]
        )

    gt = format_parses(task, parses) if reformat[0] else parses
    pr = format_parses(task, pr_parses) if reformat[1] else pr_parses

    stats = get_init_stats_task()
    corrects_parse = []

    parses_refined = []
    pr_parses_refined = []
    for pr1, gt1 in zip(pr, gt):
        if task in [
            "receipt_v1",
        ]:
            out = get_statistics_receipt_v1(
                gt1,
                pr1,
                stats=stats,
                receipt_refine=refine_parse,
                receipt_edit_distance=allow_small_edit_distance,
                return_refined_parses=return_refined_parses,
            )
        else:
            raise NotImplementedError

        if return_refined_parses:
            stats, correct, parses_refined1, pr_parses_refined1 = out
            parses_refined.append(reformat_refined(task, parses_refined1))
            pr_parses_refined.append(reformat_refined(task, pr_parses_refined1))
        else:
            stats, correct = out

        corrects_parse.append(correct)

    parse_stat, card_stat = summary_task("./result_temp.dat", stats)
    if mode != "infer":
        f1 = parse_stat["total"][-1]
    else:
        f1 = -1

    if return_refined_parses:
        return (
            f1,
            parse_stat,
            card_stat,
            corrects_parse,
            parses_refined,
            pr_parses_refined,
        )
    else:
        return f1, parse_stat, card_stat, corrects_parse


def trim_parse_for_scoring(parses, target_fields, is_shallow):
    new_parses = []
    for parse in parses:
        new_parse = []
        parse_and_other = deepcopy(parse) if is_shallow else None
        parse = deepcopy(parse if not is_shallow else parse["parse"])

        for parse1 in parse:
            if is_shallow:  # dict
                for target_field in target_fields:
                    val = parse1.pop(target_field, None)
                    if val is not None:
                        break
                if val is None:
                    new_parse.append(parse1)

            else:  # list of single dict
                if list(parse1[0])[0] not in target_fields:
                    new_parse.append(parse1)

        if is_shallow:
            parse_and_other["parse"] = new_parse
            new_parse = parse_and_other

        new_parses.append(new_parse)
    return new_parses


def format_parses(task, parses):
    if task == "namecard":
        new_parses = format_parses_namecard(parses)
    elif task in ["receipt_v1", "invoice_v2", "funsd"]:
        new_parses = format_parses_receipt_v1(parses)
    elif task in ["pubtabnet"]:
        new_parses = format_parses_receipt_v1(parses)

    elif task in ["receipt_v1_two_roots", "receipt_v1_two_roots_new_rel"]:
        # increase effective batch number
        parses_single_column = convert_parses_to_single_column(parses)
        new_parses = format_parses_receipt_v1(parses_single_column)

    else:
        raise NotImplementedError

    return new_parses


def format_parses_namecard(parses):
    outs = []
    for parse in parses:
        # if "file_id" in parse.keys():
        #     parse = parse["parse"]
        # else:
        out = {"file_id": -1}
        new_parse = []
        for parse1 in parse:
            if len(parse1) == 0:
                pass  # ficticious example due to memory limitation,
            else:
                assert len(parse1) == 1
                for k, v in parse1.items():
                    new_parse1 = {
                        "field": k,
                        "probs": [1],
                        "value": v,
                    }
                    if k != "don't care":
                        new_parse.append(new_parse1)
        out["parse"] = new_parse
        outs.append(out)

    return outs


def format_parses_receipt_v1(parses):
    outs = []
    for grouped_parse in parses:
        # if "file_id" in parse.keys():
        #     parse = parse["parse"]
        # else:
        out = {"file_id": -1}
        new_parse = []
        for grouped_parse1 in grouped_parse:
            new_parse1 = {}
            for f_parse1 in grouped_parse1:
                k = gu.get_key_from_single_key_dict(f_parse1)
                v = f_parse1[k]
                if k in new_parse1:
                    new_parse1[k] += [v]
                else:
                    new_parse1[k] = [v]
            new_parse.append(new_parse1)

        out["parse"] = new_parse
        outs.append(out)

    return outs


def convert_parses_to_single_column(parses):
    new_parses = []
    for root_parse in parses:
        new_parses += root_parse
    return new_parses


def reformat_refined(task, parse_refined):
    """t
    In:
    [ {"field": "full_name", "value": "Harry Potter"}, { ...} ]

    Out:
    [{"full_name": "Harry Potter"}, {.. }]

    """
    out = []
    for p1 in parse_refined:
        if task == "namecard":
            out1 = {p1["field"]: p1["value"]}
        else:
            out1 = p1
        out.append(out1)
    return out


def get_tp_fn_fp_edge(gt_mat, pr_mat):
    # calculate
    # tp: true positive
    # fn: false negative
    # fp: false positive
    idx_gtt = gt_mat == 1  # gt true
    idx_gtf = gt_mat == 0  # gt false
    tp1 = sum(pr_mat[idx_gtt] == 1)
    fn1 = sum(pr_mat[idx_gtt] == 0)
    fp1 = sum(pr_mat[idx_gtf] == 1)

    return tp1, fn1, fp1


def check_and_convert_list_of_tensors_to_numpy_array(list_of_tensors):
    if isinstance(list_of_tensors[0], torch.Tensor):
        return [tensor.cpu().numpy() for tensor in list_of_tensors]
    else:
        return list_of_tensors


def cal_tp_fn_fp_of_edges(labels, pr_labels):
    """
    Args:
        labels: [batch, n_relation_type, row, col]
        pr_labels: [batch, n_relation_type, row, col]

    # tp: true positive
    # fn: false negative
    # fp: false positive

    """
    labels = check_and_convert_list_of_tensors_to_numpy_array(labels)
    pr_labels = check_and_convert_list_of_tensors_to_numpy_array(pr_labels)

    n_relation_type = len(pr_labels[0])  # 2 in this study.

    tp = [0] * n_relation_type
    fn = [0] * n_relation_type
    fp = [0] * n_relation_type

    if labels[0] is not None:
        for label, pr_label in zip(labels, pr_labels):

            for i_edge_type, (label1, pr_label1) in enumerate(zip(label, pr_label)):
                gt_mat = np.array(label1)
                pr_mat = np.array(pr_label1)

                tp1, fn1, fp1 = get_tp_fn_fp_edge(gt_mat, pr_mat)

                tp[i_edge_type] += tp1
                fn[i_edge_type] += fn1
                fp[i_edge_type] += fp1

    return tp, fn, fp


def cal_p_r_f1(tp, fn, fp):
    p = []
    r = []
    f1 = []
    for tp1, fn1, fp1 in zip(tp, fn, fp):
        p1 = tp1 / (tp1 + fp1)
        r1 = tp1 / (tp1 + fn1)
        p.append(p1)
        r.append(r1)
        f1.append(2 * p1 * r1 / (p1 + r1))
    return p, r, f1


def extract_header_id_of_entity(gt_f_parse_box_id):
    header_id_of_entity = []
    header_label_of_entity = []
    for kv in gt_f_parse_box_id:
        assert len(kv) == 1
        k, v = list(kv.keys())[0], list(kv.values())[0]
        header_label_of_entity.append(k)
        header_id_of_entity.append(v[0])

    return header_id_of_entity, header_label_of_entity


def extract_header_id_of_entities(gt_f_parse_box_ids):
    header_id_of_entities = []
    header_label_of_entities = []

    for gt_f_parse_box_id in gt_f_parse_box_ids:
        (header_id_of_entity, header_label_of_entity) = extract_header_id_of_entity(
            gt_f_parse_box_id
        )

        header_id_of_entities.append(header_id_of_entity)
        header_label_of_entities.append(header_label_of_entity)

    return header_id_of_entities, header_label_of_entities


def get_headers_of(header_id_of_entities, data_ins):
    data_outs = []
    for data_in, header_id_of_entity in zip(data_ins, header_id_of_entities):
        data_outs.append(np.array(data_in)[np.array(header_id_of_entity)])
    return data_outs


def extract_links(fields, labels, target_label_id):
    links = []
    for label in labels:
        grp = np.array(label[target_label_id])
        link = extract_link(fields, grp)
        links.append(link)

    return links


def extract_link(fields, grp):
    row_offset = len(fields)
    nr, nc = grp.shape
    assert nr == nc + row_offset
    link = []
    for i_box, row in enumerate(grp[row_offset:, :]):
        i_col = np.nonzero(row == 1)[0]
        for i_col1 in i_col:
            link.append([i_box, i_col1])
    return link


def get_tp_fn_fp_link(gt_link, pr_link):
    tp = 0
    fn = 0
    fp = 0
    for gt_link1 in gt_link:
        if gt_link1 in pr_link:
            tp += 1
        else:
            fn += 1

    for pr_link1 in pr_link:
        if pr_link1 not in gt_link:
            fp += 1

    tp_fn_fp = [tp, fn, fp]

    return tp_fn_fp


def get_tp_fn_fp_all(gts, prs, fields_of_interest):
    tp_fn_fp_all = np.zeros([len(fields_of_interest), 3])
    assert len(gts) == len(prs)
    for gt, pr in zip(gts, prs):
        tp_fn_fp = get_tp_fn_fp(gt, pr, fields_of_interest)
        tp_fn_fp_all += tp_fn_fp

    return tp_fn_fp_all.tolist()


def get_tp_fn_fp(gt, pr, fields_of_interest):
    gt = np.array(gt)
    pr = np.array(pr)
    tp_fn_fp = np.zeros([len(fields_of_interest), 3])
    for i_field, field in enumerate(fields_of_interest):
        gt_sub_ids = np.nonzero(gt == field)[0]  # id for each field
        c_gt_sub_ids = np.nonzero(gt != field)[0]  # counter set
        pr_sub_ids = np.nonzero(pr == field)[0]

        # true positive
        pr_sub = pr[gt_sub_ids]
        c_pr_sub = pr[c_gt_sub_ids]
        tp = len(np.nonzero(pr_sub == field)[0])
        fn = len(np.nonzero(pr_sub != field)[0])
        fp = len(np.nonzero(c_pr_sub == field)[0])
        assert tp + fn == len(gt_sub_ids)

        tp_fn_fp[i_field, 0] = tp
        tp_fn_fp[i_field, 1] = fn
        tp_fn_fp[i_field, 2] = fp

    return tp_fn_fp.tolist()


def get_p_r_f1_entity(tp_fn_fp, fields_of_interest):
    tp_fn_fp = np.array(tp_fn_fp)
    assert (len(fields_of_interest), 3) == tp_fn_fp.shape
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 / (1/p + 1/r)
    tp = tp_fn_fp[:, 0]
    fn = tp_fn_fp[:, 1]
    fp = tp_fn_fp[:, 2]

    p_r_f1 = np.zeros([len(fields_of_interest), 3])
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)

    p_r_f1[:, 0] = p
    p_r_f1[:, 1] = r
    p_r_f1[:, 2] = f1

    tp_all, fn_all, fp_all = np.sum(tp_fn_fp, axis=0)
    p_all = tp_all / (tp_all + fp_all)
    r_all = tp_all / (tp_all + fn_all)
    f1_all = 2 * p_all * r_all / (p_all + r_all)

    p_r_f1_all = [p_all, r_all, f1_all]

    return p_r_f1.tolist(), p_r_f1_all


def get_p_r_f1_link(gt_links, pr_links):
    tp_fn_fp_all = np.zeros([3])
    for gt_link, pr_link in zip(gt_links, pr_links):
        tp_fn_fp = get_tp_fn_fp_link(gt_link, pr_link)
        tp_fn_fp_all += tp_fn_fp

    tp_a, fn_a, fp_a = tp_fn_fp_all

    p = tp_a / (tp_a + fp_a)
    r = tp_a / (tp_a + fn_a)
    f1 = 2 * p * r / (p + r)

    return [p, r, f1]


def filter_non_header_id(links, header_id_of_entities, gt):
    new_links = []
    for link, header_id_of_entity in zip(links, header_id_of_entities):
        new_link = []
        for link1 in link:
            hid, tid = link1
            if (hid in header_id_of_entity) and (tid in header_id_of_entity):
                new_link.append([hid, tid])
            else:
                assert not gt

        new_links.append(new_link)
    return new_links


def get_p_r_f1(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 / (1 / p + 1 / r)
    return p, r, f1
