# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import re
from collections import Counter

import numpy as np
from nltk.metrics.distance import edit_distance

from spade.postprocess.refine_parse_receipt_utils import refine_parse_receipt


############## RECEIPT ##############
def group_counter(group):
    txt = []
    for key in group:
        txt = txt + group[key]
    return Counter(sorted("".join(txt).replace(" ", "")))


def get_group_compare_score(gr1, gr2):
    score = 0
    # if gr1.keys() == gr2.keys():
    #    score += 100

    for key in list(set(list(gr1.keys()) + list(gr2.keys()))):
        if key in gr1 and key in gr2:
            if gr1[key] == gr2[key]:
                score += 50
            elif gr1[key] in gr2[key] or gr2[key] in gr1[key]:
                score += 30
            else:
                score += sum(
                    (Counter("".join(gr1[key])) & Counter("".join(gr2[key]))).values()
                )

    score += sum((group_counter(gr1) & group_counter(gr2)).values())
    return score


def get_int_number(string):
    num = re.sub(r"[^0-9]", "", string)
    return int(num) if len(num) > 0 else 0


def get_price_from_parse(parse):
    price = 0
    total = 0
    for gr in parse:
        for key in gr.keys():
            val = get_int_number(gr[key][0].split(" ")[0])
            if "menu.price" == key or "menu.sub_price" == key:
                price += val
            elif "menu.discountprice" == key:
                price -= val
            elif "total.total_price" == key:
                total += val

    return price, total


def get_init_stats_receipt():
    return {
        "label_stats": dict(),
        "group_stats": [0, 0, 0],
        "receipt_cnt": 0,
        "price_count_cnt": 0,
        "prices_cnt": 0,
        "receipt_total": 0,
    }


def get_statistics_receipt(
    gt,
    pr,
    stats,
    receipt_refine=False,
    receipt_edit_distance=False,
    return_refined_parses=False,
):

    gt = refine_parse_receipt(gt)["parse"] if receipt_refine else gt["parse"]
    pr = refine_parse_receipt(pr)["parse"] if receipt_refine else pr["parse"]
    label_stats = stats["label_stats"]
    group_stats = stats["group_stats"]

    mat = np.zeros((len(gt), len(pr)), dtype=np.int)
    for i, gr1 in enumerate(gt):
        for j, gr2 in enumerate(pr):
            mat[i][j] = get_group_compare_score(gr1, gr2)

    pairs = []
    gt_paired = []
    pr_paired = []
    for _ in range(min(len(gt), len(pr))):
        if np.max(mat) == 0:
            break

        x = np.argmax(mat)
        y = int(x / len(pr))
        x = int(x % len(pr))
        mat[y, :] = 0
        mat[:, x] = 0
        pairs.append((y, x))
        gt_paired.append(y)
        pr_paired.append(x)

    for i in range(len(gt)):
        stat = dict()
        for key in gt[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][1] += stat[key]

    for i in range(len(pr)):
        stat = dict()
        for key in pr[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][2] += stat[key]

    group_stat = [0, len(gt), len(pr)]
    price_count_check = True
    for i, j in pairs:
        # For each group,
        stat = dict()
        for key in set(list(gt[i].keys()) + list(pr[j].keys())):
            if key not in stat:
                stat[key] = 0

        cnt = 0
        for key in gt[i]:
            pr_val = (
                [norm_receipt(val, key) for val in pr[j][key]] if key in pr[j] else []
            )
            gt_val = [norm_receipt(val, key) for val in gt[i][key]]

            # if key in pr[j] and pr[j][key] == gt[i][key]:
            #    stat[key] += 1
            #    cnt += 1
            if pr_val == gt_val:
                stat[key] += 1
                cnt += 1

            elif (
                "nm" in key
                and receipt_edit_distance
                and len(pr_val) > 0
                and (
                    edit_distance(pr_val[0], gt_val[0]) <= 2
                    or edit_distance(pr_val[0], gt_val[0]) / len(pr_val[0]) <= 0.4
                )
            ):
                stat[key] += 1
                cnt += 1

            elif "price" in key or "cnt" in key:
                price_count_check = False

        if cnt == len(gt[i]):
            group_stat[0] += 1

        # Stat Update
        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][0] += stat[key]

    for i, gr in enumerate(gt):
        if i not in gt_paired:
            for key in gr:
                if "price" in key or "cnt" in key:
                    price_count_check = False
    for i, gr in enumerate(pr):
        if i not in pr_paired:
            for key in gr:
                if "price" in key or "cnt" in key:
                    price_count_check = False

    stats["price_count_cnt"] += price_count_check
    for k in range(3):
        group_stats[k] += group_stat[k]

    gt_prices = get_price_from_parse(gt)
    pr_prices = get_price_from_parse(pr)  # return price, total
    if gt_prices == pr_prices and gt_prices[1] != 0:  # total_price != 0
        stats["prices_cnt"] += 1

    item_correct = group_stat[0] == group_stat[1] and group_stat[1] == group_stat[2]
    stats["receipt_cnt"] += item_correct
    stats["receipt_total"] += 1

    label_stats["total"] = [0, 0, 0]
    for key in sorted(label_stats):
        if key not in ["total"]:
            for i in range(3):
                label_stats["total"][i] += label_stats[key][i]

    stats["label_stats"] = label_stats
    stats["group_stats"] = group_stats
    if return_refined_parses:
        return stats, item_correct, gt, pr
    else:
        return stats, item_correct


def norm_receipt(val, key):
    val = val.replace(" ", "")
    return val


def get_scores(tp, fp, fn):
    pr = tp / (tp + fp) if (tp + fp) != 0 else 0
    re = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * pr * re) / (pr + re) if (pr + re) != 0 else 0
    return pr, re, f1


def summary_receipt(path, stats, print_screen=False):
    st = stats["label_stats"]
    st["Group_accuracy"] = stats["group_stats"]

    s = dict()
    for key in st:
        tp = st[key][0]
        fp = st[key][2] - tp
        fn = st[key][1] - tp
        s[key] = (tp, fp, fn) + get_scores(tp, fp, fn)

    c = {
        "main_key": "receipt",
        "prices": stats["prices_cnt"],
        "price/cnt": stats["price_count_cnt"],
        "receipt": stats["receipt_cnt"],
        "total": stats["receipt_total"],
    }

    if print_screen:
        other_fields = ["total", "Group_accuracy"]
        header = ("field", "tp", "fp", "fn", "prec", "rec", "f1")
        print("%25s\t%6s\t%6s\t%6s\t%6s\t%6s\t%6s" % header)
        print(
            "------------------------------------------------------------------------------"
        )
        for key in sorted(s):
            if key not in other_fields:
                print(
                    "%-25s\t%6d\t%6d\t%6d\t%6.3f\t%6.3f\t%6.3f"
                    % (
                        key,
                        s[key][0],
                        s[key][1],
                        s[key][2],
                        s[key][3],
                        s[key][4],
                        s[key][5],
                    )
                )
        print(
            "------------------------------------------------------------------------------"
        )
        for key in other_fields:
            print(
                "%-25s\t%6d\t%6d\t%6d\t%6.3f\t%6.3f\t%6.3f"
                % (
                    key,
                    s[key][0],
                    s[key][1],
                    s[key][2],
                    s[key][3],
                    s[key][4],
                    s[key][5],
                )
            )

        for key in c:
            if key not in ["total", "main_key"]:
                print(
                    " - %10s accuracy :  %.4f (%d/%d)"
                    % (key, c[key] / c["total"], c[key], c["total"])
                )

    return s, c
