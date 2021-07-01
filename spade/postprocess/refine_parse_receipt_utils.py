# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import re
from copy import deepcopy


def refine_ind_text(key, val, idx):
    if "price" in key:
        val = ("-" if val.startswith("-") and idx == 0 else "") + re.sub(
            r"[^0-9]+", "", val
        )

    elif "cnt" in key:
        val = ("-" if val.startswith("-") and idx == 0 else "") + re.sub(
            r"[^0-9^\.]+", "", val
        )

    return val


def refine_parse_receipt(parse_orig):
    parse = parse_orig["parse"]
    parse_refined = deepcopy(parse_orig)
    if "prob" in parse_orig:
        prob = parse_orig["prob"]
        flag_prob_exist = True
    else:
        flag_prob_exist = False
        prob = None

    if "lang" not in parse_orig or parse_orig["lang"] == "ind":
        refine_text = refine_ind_text
    else:
        raise NotImplementedError

    for i, group_parse in enumerate(parse):
        # To handle the edge case: (menu.unitprice without menu.price & menu.itemsubtotal)
        keys = group_parse.keys()
        if (
            "menu.unitprice" in keys
            and "menu.itemsubtotal" not in keys
            and "menu.price" not in keys
        ):
            parse[i]["menu.price"] = parse[i].pop("menu.unitprice")
            if flag_prob_exist:
                if prob[i]:
                    prob[i]["menu.price"] = prob[i].pop("menu.unitprice")

        for key in group_parse:
            if "unitprice" in key and len(group_parse[key]) > 1:
                group_parse[key] = ["".join(group_parse[key])]

            for j, val in enumerate(group_parse[key]):
                val = refine_text(key, val, j)
                parse[i][key][j] = val

    # Return
    parse_refined["parse"] = parse
    parse_refined["prob"] = prob
    return parse_refined
