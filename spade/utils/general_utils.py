# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import json
import multiprocessing
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import torch
import yaml


def remove_duplicate_in_1d_list(data_in: list):
    assert type(data_in) == list
    return list(OrderedDict.fromkeys(data_in))


def write_json(path, out1):
    with open(path, "wt", encoding="utf-8") as f:
        json_str = json.dumps(out1, ensure_ascii=False)
        json_str += "\n"
        f.writelines(json_str)


def write_jsonl(path, out):
    with open(path, "wt", encoding="utf-8") as f:
        for out1 in out:
            json_str = json.dumps(out1, ensure_ascii=False)
            json_str += "\n"
            f.writelines(json_str)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
        data = json.loads(data.strip())

    return data


def load_yaml(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.full_load(f)

    return data


def load_jsonl(filepath, toy_data=False, toy_size=4, shuffle=False):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size:
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print("The data shuffled.")
        seed = 1
        random.Random(seed).shuffle(data)  # fixed

    return data


def gen_backbone_path(path_data_folder, backbone_name, backbone_tweak_tag):
    return (
        Path(path_data_folder)
        / "model"
        / "backbones"
        / backbone_name
        / backbone_tweak_tag
    )


def cnt_model_weights(model):
    nw = 0
    for param in model.parameters():
        nw += torch.prod(torch.tensor(param.shape))

    print(f"The # of weights {nw:.3g}")


def update_part_of_model(parent_model_state_dict, child_model, rank):
    child_model_dict = child_model.state_dict()

    # 1. filter out unnecessary keys
    # partial_parent_model_dict = {k: v for k, v in parent_model_dict.items() if k in child_model_dict}
    partial_parent_model_dict = {}
    for k, v in parent_model_state_dict.items():
        if k in child_model_dict:
            if rank == 0:
                print(f"{k} updated")
            v_child = child_model_dict[k]
            if v.shape == v_child.shape:
                partial_parent_model_dict[k] = v
            else:
                if rank == 0:
                    print(
                        f"!!!!{k} param shows size mismatch between parent and child models!!!!"
                    )

        else:
            if rank == 0:
                print(f"!!!!{k} model param. is not presented in child model!!!!")

    # omitted_key = [k for k, v in parent_model_dict.items() if k not in child_model_dict]
    # print(f"Omittted keys: {omitted_key}")

    # 2. overwrite entries in the existing state dict
    child_model_dict.update(partial_parent_model_dict)

    child_model.load_state_dict(child_model_dict)

    return child_model


def gen_slices(dim, i, j):
    _s = slice(i, j)
    _ss = [_s] * dim
    return _ss


def get_key_from_single_key_dict(f_parse1):
    target_field_list = list(f_parse1.keys())
    assert len(target_field_list) == 1
    field_of_target = target_field_list[0]

    return field_of_target


def get_local_rank():
    """
    Pytorch lightning save local rank to environment variable "LOCAL_RANK".
    From rank_zero_only
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank


def save_pytorch_model(path_save_dir, model):
    state = {"state_dict": model.state_dict()}
    torch.save(state, os.path.join(path_save_dir, "model.pt"))


def get_char_for_detokenization(backbone_name):
    if backbone_name in ["bert-base-multilingual-cased"]:
        return "#"
    else:
        raise NotImplementedError


def timeit(fun):
    def timed(*args, **kw):
        st = time.time()
        result = fun(*args, **kw)
        ed = time.time()
        print(f"Execution time of {fun.__name__} = {ed - st}s")
        return result

    return timed
