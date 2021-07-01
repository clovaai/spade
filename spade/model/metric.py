# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch
import torchmetrics

import spade.utils.analysis_utils as au
import spade.utils.general_utils as gu


class SpadeMetric(torchmetrics.Metric):
    def __init__(self, n_relation_type, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # self.add_state("f1", default=torch.FloatTensor(0), dist_reduce_fx="mean")  # parse
        self.add_state(
            "tp_edge", default=torch.zeros(n_relation_type), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp_edge", default=torch.zeros(n_relation_type), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn_edge", default=torch.zeros(n_relation_type), dist_reduce_fx="sum"
        )
        self.add_state("tp_parse", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("fp_parse", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("fn_parse", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse):
        self.tp_edge += torch.tensor(tp_edge).type_as(self.tp_edge)
        self.fp_edge += torch.tensor(fp_edge).type_as(self.fp_edge)
        self.fn_edge += torch.tensor(fn_edge).type_as(self.fn_edge)
        self.tp_parse += torch.tensor(tp_parse).type_as(self.tp_parse)
        self.fp_parse += torch.tensor(fp_parse).type_as(self.fp_parse)
        self.fn_parse += torch.tensor(fn_parse).type_as(self.fn_parse)

    def compute(self):
        p_parse, r_parse, f1_parse = au.get_p_r_f1(
            self.tp_parse, self.fp_parse, self.fn_parse
        )
        p_edge_avg, r_edge_avg, f1_edge_avg = au.get_p_r_f1(
            self.tp_edge, self.fp_edge, self.fn_edge
        )

        return p_edge_avg, r_edge_avg, f1_edge_avg, p_parse, r_parse, f1_parse
