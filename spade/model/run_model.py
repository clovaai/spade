# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import os
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import torch

import spade.utils.general_utils as gu
from spade.model.data_module import SpadeDataModule
from spade.model.model import RelationTagger


def prepare_data_model_trainer(cfg):
    path_logs = Path("./logs") / cfg.config_file_name
    os.makedirs(path_logs, exist_ok=True)
    tb_logger = pl.loggers.TensorBoardLogger(path_logs)

    data_module = SpadeDataModule(cfg)
    model = get_model(
        cfg.model_param,
        cfg.train_param,
        cfg.infer_param,
        cfg.path_data_folder,
        cfg.verbose,
    )

    if cfg.model_param.weights.trained:
        # old_format = not False
        old_format = False
        if old_format:
            snapshot = torch.load(cfg.model_param.path_trained_model)
            new_snapshot = {}
            new_snapshot["state_dict"] = snapshot["model"]
            torch.save(new_snapshot, cfg.model_param.path_trained_model)

        model = model.load_from_checkpoint(
            cfg.model_param.path_trained_model,
            hparam=cfg.model_param,
            tparam=cfg.train_param,
            iparam=cfg.infer_param,
            path_data_folder=cfg.path_data_folder,
            verbose=cfg.verbose,
        )
        print(f"The trained model is loaded from {cfg.model_param.path_trained_model}")

    trainer = pl.Trainer(
        logger=tb_logger,
        log_every_n_steps=cfg.train_param.get("log_every_n_steps", 50),
        gpus=torch.cuda.device_count(),
        max_epochs=cfg.train_param.max_epochs,
        val_check_interval=cfg.train_param.val_check_interval,
        limit_train_batches=cfg.train_param.limit_train_batches,
        limit_val_batches=cfg.train_param.limit_val_batches,
        num_sanity_val_steps=1,
        progress_bar_refresh_rate=100,
        accumulate_grad_batches=cfg.train_param.accumulate_grad_batches,
        accelerator=cfg.train_param.accelerator,
        precision=cfg.model_param.precision,
        gradient_clip_val=cfg.train_param.gradient_clip_val,
        gradient_clip_algorithm=cfg.train_param.gradient_clip_algorithm,
    )

    return data_module, model, trainer


def get_model(hparam, tparam, iparam, path_data_folder, verbose=False):
    if hparam.model_name == "RelationTagging":
        model = RelationTagger(
            hparam, tparam, iparam, path_data_folder, verbose=verbose
        )
    else:
        raise NotImplementedError

    return model


def do_training(cfg):
    data_module, model, trainer = prepare_data_model_trainer(cfg)
    trainer.fit(model, data_module)


def do_testing(cfg):
    data_module, model, trainer = prepare_data_model_trainer(cfg)
    trainer.test(model, datamodule=data_module)


def do_prediction(cfg, path_predict_input_json):
    data_module, model, trainer = prepare_data_model_trainer(cfg)
    assert cfg.raw_data_input_type == "type1"

    data_module.path_predict_input_json = path_predict_input_json
    out = trainer.predict(model, datamodule=data_module)[0]
    path_to_save = path_predict_input_json.__str__() + ".out.json"
    gu.write_json(path_to_save, out)
    pprint(out["pr_parse"])
