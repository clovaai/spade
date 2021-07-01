# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch
import transformers


def get_optimizer(tparam, model):
    _lr_type, lr_param = get_lr_type_and_param(tparam)
    map_optimizers_name_to_type = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }
    optimizer_type = map_optimizers_name_to_type[tparam.optimizer_type]

    if model.name == "RelationTagging":
        param_list = [
            {
                "params": filter(
                    lambda p: p.requires_grad, model.encoder_layer.parameters()
                ),
                "lr": lr_param.lr_enc,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad, model.decoder_layer.parameters()
                ),
                "lr": lr_param.lr_dec,
            },
        ]

        optimizer = optimizer_type(param_list, lr=lr_param.lr_default, weight_decay=0)

    else:
        raise NotImplementedError

    return optimizer


def get_lr_type_and_param(tparam):
    lr_type = tparam.lr_scheduler_type
    lr_param = tparam.lr_scheduler_param[lr_type]
    return lr_type, lr_param


def gen_lr_scheduler(tparam, optimizer, lr_type, lr_param):
    if lr_type == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda epoch: 1, lambda epoch: 1], verbose=True
        )
    elif lr_type == "multi_step_lr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_param["milestones"],
            gamma=lr_param["gamma"],
            verbose=True,
        )

    elif lr_type == "warmup_constant":
        lr_scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=lr_param.num_warmup_steps
        )
    elif lr_type == "cos_with_hard_restarts":
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=lr_param.num_warmup_steps,
            num_training_steps=lr_param.num_training_steps,
            num_cycles=lr_param.num_cycles,
        )
    elif lr_type == "linear":
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=lr_param.num_warmup_steps,
            num_training_steps=tparam.max_epochs,
        )

    else:
        raise NotImplementedError
    return lr_scheduler


def get_lr_dict(optimizer, tparam):
    lr_type, lr_param = get_lr_type_and_param(tparam)
    lr_scheduler = gen_lr_scheduler(tparam, optimizer, lr_type, lr_param)
    lr_dict = {
        "scheduler": lr_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "val_loss",
        "strict": True,
        "name": None,
    }

    return lr_dict
