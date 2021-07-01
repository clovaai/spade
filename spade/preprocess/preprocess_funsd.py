# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import os
from pathlib import Path

import cv2

import spade.utils.general_utils as gu


def get_filepaths(load_dir, file_extension):
    fnames = os.listdir(load_dir)
    filepaths = [os.path.join(load_dir, f) for f in fnames]
    # files = filter(os.path.isfile, files)
    filepaths = [x for x in filepaths if x.endswith(file_extension)]
    return fnames, filepaths


def gen_data(mode):
    assert mode in ["train", "test"]
    data_dir = Path("./data") / "funsd"
    path_ori0 = data_dir / "original" / "dataset" / f"{mode}ing_data"

    path_ori_json = path_ori0 / "annotations"
    path_ori_img = path_ori0 / "images"

    fnames_json, filepaths_json = get_filepaths(path_ori_json, ".json")

    ##
    new_data = []
    for fname, fpath in zip(fnames_json, filepaths_json):
        t1 = gu.load_json(fpath)
        t1["meta"] = {}
        t1["meta"]["fname"] = fname
        image_id = fname.split(".")[0]
        t1["meta"]["image_id"] = image_id

        fpath_img = path_ori_img / f"{image_id}.png"

        img = cv2.imread(fpath_img.__str__())
        width, height = img.shape[1], img.shape[0]
        t1["meta"]["image_size"] = {"width": width, "height": height}
        new_data.append(t1)

    path_save = data_dir / mode / f"{mode}_type0.jsonl"
    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    gu.write_jsonl(path_save, new_data)
    if mode == "train":
        # also generate dummy dev_type0.jsonl
        path_save = data_dir / "dev" / f"dev_type0.jsonl"
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        gu.write_jsonl(path_save, new_data[:8])

    return None


def run_preprocess_funsd():
    for mode in ["test", "train"]:
        gen_data(mode)
