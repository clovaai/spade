# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

from spade.model.data_module import SpadeDataModule
from spade.preprocess.preprocess_funsd import run_preprocess_funsd


def do_preprocess(cfg):
    if cfg.model_param.task == "funsd":
        # download and combine the original data preprocess
        run_preprocess_funsd()
    data_module = SpadeDataModule(cfg)
    train_data, dev_data = data_module._prepare_train_data()
    datas = data_module._prepare_test_datas()

    datas = datas[1:]  # remove dev_data
    datas = [train_data, dev_data] + list(datas)  # combine all
    for data in datas:
        data.gen_type1_data()

    pass
