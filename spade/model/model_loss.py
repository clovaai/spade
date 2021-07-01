# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import numpy as np
import torch
import torch.nn.functional as F


def Loss_rt(score, label, n_fields, l_units, loss_weights):
    # label = [batch_size, n_relation, nr, nc]. [list, list, 2darray]
    loss = 0
    batch_size, n_relation_doubled, nr, nc = score.shape
    n_relation = int(n_relation_doubled / 2)
    for b in range(batch_size):
        nc1 = l_units[b]
        nr1 = nc1 + n_fields
        for i_rel in range(n_relation):
            label1 = label[b][i_rel][:nr1, :nc1]
            st_rel = 2 * i_rel
            _label = label1.unsqueeze(0)
            loss += F.cross_entropy(
                score[b : b + 1, st_rel : (st_rel + 2), :nr1, :nc1],
                _label,
                weight=loss_weights.type_as(score),
            )

    return loss
