# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import spade.model.model_utils as mu


class SpadeDecoder(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_relation_type,
        fields,
        token_lv_boxing,
        include_second_order_relations,
        vi_params=None,
    ):
        super().__init__()

        # define metric for relation tagging
        self.n_relation_type = n_relation_type
        self.n_fields = len(fields)

        self.token_lv_boxing = token_lv_boxing
        self.h_pooler = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for _ in range(n_relation_type)]
        )
        self.t_pooler = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for _ in range(n_relation_type)]
        )
        self.W_label = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False)
                for _ in range(2 * (n_relation_type))
            ]
            # include edge (+1). 0-score and 1-score (*2)
        )

        # define embedding for each fields and [end] special tokens
        self.sph_emb = nn.ModuleList(
            [nn.Embedding(self.n_fields, hidden_size) for _ in range(n_relation_type)]
        )

        self.include_second_order_relations = include_second_order_relations
        if include_second_order_relations:
            self.n_vi_iter = vi_params["n_vi_iter"]
            self.do_gp = vi_params["do_gp"]
            self.do_sb = vi_params["do_sb"]

            self.ht_pooler = nn.ModuleList(
                [nn.Linear(input_size, hidden_size) for _ in range(n_relation_type)]
            )
            self.W_uni = nn.ModuleList(
                [
                    nn.Linear(hidden_size, hidden_size, bias=False)
                    for _ in range(1 * (n_relation_type))
                ]
            )
            # grand parents
            self.W_gp = nn.ModuleList(
                [
                    nn.Linear(hidden_size, hidden_size, bias=False)
                    for _ in range(3 * (n_relation_type))
                ]
            )
            # sibling
            self.W_sb = nn.ModuleList(
                [
                    nn.Linear(hidden_size, hidden_size, bias=False)
                    for _ in range(3 * (n_relation_type))
                ]
            )

        # special token header embedding
        self.initializer_range = 0.02  # std

    def forward(self, encoded, header_ids, lmax_boxes):

        batch_size, input_len, idim = encoded.shape
        l_boxes = header_ids.sum(dim=1)

        enc_header = mu.select_header_vec(
            batch_size,
            lmax_boxes,
            header_ids,
            l_boxes,
            idim,
            encoded,
            self.token_lv_boxing,
        )

        lmax_units = input_len if self.token_lv_boxing else lmax_boxes

        # get score
        if self.include_second_order_relations:
            # initialize score
            unary_score = torch.zeros(
                [
                    batch_size,
                    self.n_relation_type,
                    2,  # Z = 0 or 1
                    lmax_units + self.n_fields,
                    lmax_units,
                ]
            )

            score = torch.zeros(
                [
                    batch_size,
                    self.n_relation_type,  # +1 for edge
                    2,  # Z = 0 or 1
                    lmax_units + self.n_fields,
                    lmax_units,
                ]
            )

            ternary_score_gp = torch.zeros(
                [
                    batch_size,
                    self.n_relation_type,
                    lmax_units + self.n_fields,
                    lmax_units,
                    lmax_units,
                ]
            )

            ternary_score_sb = torch.zeros(
                [
                    batch_size,
                    self.n_relation_type,
                    lmax_units + self.n_fields,
                    lmax_units,
                    lmax_units,
                ]
            )

            # calculate score.
            for i_label in range(self.n_relation_type):  # +1 for edge
                enc_header_h = self.h_pooler[i_label](enc_header)
                enc_header_t = self.t_pooler[i_label](enc_header)
                enc_header_ht = self.ht_pooler[i_label](enc_header)
                enc_sp = mu.embed_fields(
                    self.sph_emb[i_label], self.n_fields, batch_size
                )  # [bS, n_sp, dim]
                enc_header_h_all = torch.cat([enc_sp, enc_header_h], dim=1)

                unary_score[:, i_label, 1, :, :] = torch.matmul(
                    enc_header_h_all,  # [batch, n_field + len_box, dim]
                    self.W_uni[i_label](enc_header_t).transpose(1, 2),
                )

                # second order score
                if self.do_gp[i_label]:
                    # grand parents
                    # s_ij,jk : i-> j -? k

                    g0_gp = self._gen_g_vector(self.W_gp, 3 * i_label, enc_header_h_all)
                    g1_gp = self._gen_g_vector(
                        self.W_gp, 3 * i_label + 1, enc_header_ht
                    )
                    g2_gp = self._gen_g_vector(self.W_gp, 3 * i_label + 2, enc_header_t)

                    ternary_score_gp[:, i_label, :, :, :] = torch.einsum(
                        "bid,bjd,bkd->bijk", g0_gp, g1_gp, g2_gp
                    )

                if self.do_sb[i_label]:
                    # sibling
                    # s_ij,ik: i->j, i->k
                    g0_sb = self._gen_g_vector(self.W_sb, 3 * i_label, enc_header_h_all)
                    g1_sb = self._gen_g_vector(self.W_sb, 3 * i_label + 1, enc_header_t)
                    g2_sb = self._gen_g_vector(self.W_sb, 3 * i_label + 2, enc_header_t)

                    ternary_score_sb[:, i_label, :, :, :] = torch.einsum(
                        "bid,bjd,bkd->bijk", g0_sb, g1_sb, g2_sb
                    )

            # VI now
            score[:] = unary_score[:]
            for i_vi in range(self.n_vi_iter):
                q_value = F.softmax(score, dim=2)
                # calculate F for Z=1 case

                # get F value
                F_value = self.get_F_value(
                    q_value,
                    ternary_score_sb,
                    ternary_score_gp,
                    self.n_fields,
                    self.do_sb,
                    self.do_gp,
                )
                # update Q
                # score[:, :, 0, :, :] = 0
                score = unary_score + F_value

            # reshape q_value for the consistency with zeroth-order
            score = score.view(
                batch_size,
                2 * (self.n_relation_type),
                lmax_units + self.n_fields,
                lmax_units,
            )

        else:
            score = torch.zeros(
                [
                    batch_size,
                    2 * (self.n_relation_type),  # +1 for edge
                    lmax_units + self.n_fields,
                    lmax_units,
                ]
            ).type_as(enc_header)

            for i_label in range(self.n_relation_type):  # +1 for edge
                enc_header_h = self.h_pooler[i_label](enc_header)
                enc_header_t = self.t_pooler[i_label](enc_header)
                enc_sp = mu.embed_fields(
                    self.sph_emb[i_label], self.n_fields, batch_size
                )  # [bS, n_sp, dim]
                score[:, 2 * i_label, :, :] = torch.matmul(
                    torch.cat(
                        [enc_sp, enc_header_h], dim=1
                    ),  # [batch, n_field + len_box, dim]
                    self.W_label[2 * i_label](enc_header_t).transpose(1, 2),
                )

                score[:, 2 * i_label + 1, :, :] = torch.matmul(
                    torch.cat(
                        [enc_sp, enc_header_h], dim=1
                    ),  # [batch, n_field + len_box, dim]
                    self.W_label[2 * i_label + 1](enc_header_t).transpose(1, 2),
                )

        return score

    @staticmethod
    def _gen_g_vector(W, i_type, vec):
        return W[i_type](vec)

    @staticmethod
    def get_F_value(
        q_value, ternary_score_sb, ternary_score_gp, n_fields, do_sb, do_gp
    ):
        batch_size, n_edge_type, n_cases, n_row, n_col = q_value.shape
        assert n_cases == 2

        if sum(do_gp) == n_edge_type and sum(do_sb) == n_edge_type:
            F_value_sb = torch.einsum(
                "bnik,bnijk->bnij", q_value[:, :, 1, :, :], ternary_score_sb
            )

            F_value_gpA = torch.einsum(
                "bnjk,bnijk->bnij", q_value[:, :, 1, n_fields:, :], ternary_score_gp
            )

            F_value_gpB = torch.zeros([batch_size, n_edge_type, n_row, n_col])
            F_value_gpB[:, :, n_fields:, :] = torch.einsum(
                "bnki,bnkij->bnij", q_value[:, :, 1, :, :], ternary_score_gp
            )

            F_value = F_value_sb + F_value_gpA + F_value_gpB
        else:
            F_value = torch.zeros([batch_size, n_edge_type, n_row, n_col])
            for i_label in range(n_edge_type):
                if do_sb[i_label]:
                    F_value_sb1 = torch.einsum(
                        "bik,bijk->bij",
                        q_value[:, i_label, 1, :, :],
                        ternary_score_sb[:, i_label, :, :, :],
                    )
                    F_value[:, i_label, :, :] = F_value[:, i_label, :, :] + F_value_sb1

                if do_gp[i_label]:
                    F_value_gpA1 = torch.einsum(
                        "bjk,bijk->bij",
                        q_value[:, i_label, 1, n_fields:, :],
                        ternary_score_gp[:, i_label, :, :, :],
                    )

                    F_value_gpB1 = torch.zeros([batch_size, n_row, n_col])
                    F_value_gpB1[:, n_fields:, :] = torch.einsum(
                        "bki,bkij->bij",
                        q_value[:, i_label, 1, :, :],
                        ternary_score_gp[:, i_label, :, :, :],
                    )

                    F_value[:, i_label, :, :] = F_value[:, i_label, :, :] + F_value_gpA1
                    F_value[:, i_label, :, :] = F_value[:, i_label, :, :] + F_value_gpB1

        zero_value = torch.zeros_like(F_value)
        F_value = torch.cat([zero_value.unsqueeze(2), F_value.unsqueeze(2)], dim=2)

        return F_value
