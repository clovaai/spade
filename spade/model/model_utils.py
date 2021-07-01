# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import math
import os
from collections import Iterable
from copy import deepcopy
from functools import reduce
from pathlib import Path

import numpy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers

import spade.utils.analysis_utils as au
import spade.utils.general_utils as gu


def get_tokenizer(path_data_folder, backbone_model_name, backbone_tweak_tag):
    backbone_path = gu.gen_backbone_path(
        path_data_folder, backbone_model_name, backbone_tweak_tag
    )
    vocab_path = Path(backbone_path) / "vocab.txt"
    # if backbone_model_name == "multi_cased_L-12_H-768_A-12":
    #     tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
    if backbone_model_name == "bert-base-multilingual-cased":
        tokenizer = transformers.BertTokenizer(
            vocab_file=vocab_path, do_lower_case=False
        )
    else:
        raise NotImplementedError

    return tokenizer


def get_pretrained_transformer(
    path_data_folder, backbone_model_name, backbone_tweak_tag
):
    backbone_path = gu.gen_backbone_path(
        path_data_folder, backbone_model_name, backbone_tweak_tag
    )
    path_pretrained_transformer = backbone_path

    if backbone_model_name in ["bert-base-multilingual-cased"]:
        func_config = transformers.BertConfig
        func_model = transformers.BertModel
    elif backbone_model_name in ["xlm-roberta-base_lm", "xlm-roberta-large_lm"]:
        func_config = transformers.XLMRobertaConfig
        func_model = transformers.XLMRobertaForCausalLM
    elif backbone_model_name in ["facebook/mbart-large-cc25"]:
        func_config = transformers.MBartConfig
        func_model = transformers.MBartForConditionalGeneration
    else:
        raise NotImplementedError

    pretrained_transformer_config = func_config.from_pretrained(
        path_pretrained_transformer
    )
    pretrained_transformer = func_model.from_pretrained(path_pretrained_transformer)
    print(f"The weight of the pretraiend model {backbone_model_name}")
    gu.cnt_model_weights(pretrained_transformer)

    return pretrained_transformer, pretrained_transformer_config


def check_consistency_between_backbone_and_encoder(
    pretrained_transformer_cfg, transformer_cfg
):
    assert pretrained_transformer_cfg.hidden_size == transformer_cfg.hidden_size
    assert (
        pretrained_transformer_cfg.intermediate_size
        == transformer_cfg.intermediate_size
    )
    assert pretrained_transformer_cfg.pooler_fc_size == transformer_cfg.pooler_fc_size
    assert (
        pretrained_transformer_cfg.num_attention_heads
        == transformer_cfg.num_attention_heads
    )

    assert (
        pretrained_transformer_cfg.max_position_embeddings
        == transformer_cfg.max_position_embeddings
    )
    assert pretrained_transformer_cfg.hidden_act == transformer_cfg.hidden_act
    assert pretrained_transformer_cfg.architectures == transformer_cfg.architectures
    return None


class SinCosPositionalEncoding(pl.LightningModule):
    def __init__(self, dim):
        # ws: Same as Transformer inv_freq = 1 / 10000^(-j / dim) . why log?
        super().__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape  # [B, T]
        x = input.view(-1).float()
        y = self.inv_freq

        sinusoid_in = torch.ger(x, y)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb.detach()


def select_header_vec(
    batch_size, lmax_boxes, header_ids, l_boxes, idim, encoded, token_lv_boxing
):
    if token_lv_boxing:
        encoded_header = encoded
    else:
        encoded_header = torch.zeros([batch_size, lmax_boxes, idim]).type_as(encoded)

        for b in range(batch_size):
            # extract header vector
            idx_header = torch.nonzero(header_ids[b])  # [N, 1]
            idx_header = idx_header.squeeze(1)
            header_vec1 = encoded[b].index_select(0, idx_header)  # [n_header, dim]

            # fill concatenated header1 into sequence_output
            encoded_header[b, : l_boxes[b]] = header_vec1
    return encoded_header


def embed_fields(sph_emb, n_fields, batch_size):
    _ids = torch.arange(0, n_fields).type_as(sph_emb.weight)
    _ids = _ids.type(torch.long)
    enc_sp1 = sph_emb(_ids).unsqueeze(0)  # [1, n_sp, dim]
    enc_sp = enc_sp1.expand(batch_size, -1, -1)  # [bS, n_sp, dim]
    return enc_sp


def gen_split(feature, i_tok1, j_tok1, max_input_len):
    # F.pad(a, [0, 2, 0, 2])  row-left, row-right, col-left, col-right
    dim = feature.dim()
    split = torch.zeros(dim * [max_input_len]).type_as(feature)
    _ss0 = gu.gen_slices(dim, 0, j_tok1 - i_tok1)
    _ss = gu.gen_slices(dim, i_tok1, j_tok1)
    split[tuple(_ss0)] = feature[tuple(_ss)]

    return split


def gen_l_mask(i_sep, i_toks, j_toks):
    batch_size = len(i_toks)
    l_mask = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        i_tok1 = i_toks[b][i_sep]
        j_tok1 = j_toks[b][i_sep]
        l_mask[b] = j_tok1 - i_tok1

    return l_mask


def collect_output_of_target_layer(
    all_encoder_layer, encoder_layer_ids_used_in_decoder
):
    encoded1 = []
    for i in encoder_layer_ids_used_in_decoder:
        encoded1.append(all_encoder_layer[i])

    encoded1 = torch.cat(encoded1, dim=-1)  # [batch_size, max_input_len, dim]

    return encoded1


def get_split_param1(text_tok, max_input_len, input_split_overlap_len, do_split=True):
    l_tok = len(text_tok)

    if not do_split:
        n_sep = 1
        i_tok = [0]
        j_tok = [l_tok]

    else:
        step_size = max_input_len - input_split_overlap_len
        if l_tok <= max_input_len:
            n_sep = 1
        else:
            n_sep = int(math.ceil((l_tok - max_input_len) / step_size)) + 1
        ed_arr = np.arange(max_input_len, l_tok, step_size, dtype=np.int).tolist()
        ed_arr.append(l_tok)
        assert n_sep == len(ed_arr)

        i_tok = [0]
        j_tok = []

        for i_sep in range(n_sep - 1):
            ed1 = ed_arr[i_sep]
            j_tok.append(ed1)

            st1 = ed_arr[i_sep + 1] - max_input_len
            i_tok.append(st1)

        j_tok.append(l_tok)
    return n_sep, i_tok, j_tok, l_tok


def separate_center_coord_to_xy(center_toks):
    # [batch_size, l_tok, l_tok, 2]
    # list of 3d arrays.
    x = []
    y = []
    for arr in center_toks:
        x.append(arr[:, :, 0])
        y.append(arr[:, :, 1])
    return x, y


def convert_feature_to_tensor(lmax_tok, *args):
    # Make one long tensor.
    # Separate them inside of model.forward (and run as many as needed)
    # Concenptually simpler and more compatible.

    nargs = []
    for ids in args:
        nargs.append(pad_ids(ids, max_length=lmax_tok))

    return nargs


def convert_split_params_to_tensor(n_seps, i_toks, j_toks, l_toks):
    nmax = max(n_seps)
    # l_toks = torch.tensor(l_toks, dtype=torch.long)
    # n_seps = torch.tensor(n_seps, dtype=torch.long)
    (i_toks, j_toks) = convert_feature_to_tensor(nmax, i_toks, j_toks)

    return n_seps, i_toks, j_toks, l_toks


def pad_ids(ids, max_length=None, type_informer_tensor=None):
    """
    zero-pad. Convert list to tensor
    """
    # size
    if type_informer_tensor is None:
        assert isinstance(ids[0], torch.Tensor)
        type_informer_tensor = ids[0]
    batch_size = len(ids)
    l = [len(id) for id in ids]
    if max_length is None:
        max_length = max(l)

    # array
    dim = get_dim_of_id(ids[0])
    if isinstance(ids[0], torch.Tensor):
        my_zeros = torch.zeros
        my_long = torch.long
    else:
        my_zeros = np.zeros
        my_long = np.long

    ids_tensor = my_zeros([batch_size] + dim * [max_length], dtype=my_long)

    for b, l1 in enumerate(l):
        _ss = gu.gen_slices(dim, 0, l1)
        _ss = [b] + _ss
        ids_tensor[tuple(_ss)] = ids[b]

    # tensor
    if isinstance(ids_tensor, np.ndarray):
        ids_tensor = torch.tensor(ids_tensor)
    return ids_tensor.type_as(type_informer_tensor)


def get_dim_of_id(id):
    if isinstance(id[0], Iterable):
        # id = np.array(id)
        dim = len(id.shape)
        for d1 in id.shape:
            assert (
                d1 == id.shape[0]
            )  # assume the length in each dimension is identical.
    else:
        dim = 1

    return dim


def collect_features_batchwise(features):
    data_ids = [x["data_id"] for x in features]
    image_urls = [x["image_url"] for x in features]
    texts = [x["text"] for x in features]
    text_toks = [x["text_tok"] for x in features]
    text_tok_ids = [x["text_tok_id"] for x in features]
    labels = [x["label"] for x in features]
    label_toks = [x["label_tok"] for x in features]
    rn_center_toks = [x["rn_center_tok"] for x in features]
    rn_dist_toks = [x["rn_dist_tok"] for x in features]
    rn_angle_toks = [x["rn_angle_tok"] for x in features]
    vertical_toks = [x["vertical_tok"] for x in features]
    char_size_toks = [x["char_size_tok"] for x in features]
    header_toks = [x["header_tok"] for x in features]

    return (
        data_ids,
        image_urls,
        texts,
        text_toks,
        text_tok_ids,
        labels,
        label_toks,
        rn_center_toks,
        rn_dist_toks,
        rn_angle_toks,
        vertical_toks,
        char_size_toks,
        header_toks,
    )


class RelationTaggerUtils:
    @classmethod
    def split_features(cls, n_seps, i_toks, j_toks, max_input_len, *featuress):
        n_sep_max = max(n_seps)
        splitsss = []  # [feature, n_sep, batch_size]
        for features in featuress:
            splitss = []  # [n_sep, batch_size]

            for i_sep in range(n_sep_max):
                splits = []
                for b, feature in enumerate(features):
                    i_tok1 = i_toks[b][i_sep]
                    j_tok1 = j_toks[b][i_sep]

                    split = gen_split(feature, i_tok1, j_tok1, max_input_len)
                    splits.append(split)

                splits = torch.stack(splits, dim=0)
                splitss.append(splits)
            splitsss.append(splitss)

        return splitsss

    @classmethod
    def gen_input_mask(cls, i_sep, l_toks, i_toks, j_toks, max_input_len):
        batch_size = len(l_toks)
        l_mask = gen_l_mask(i_sep, i_toks, j_toks)

        attention_mask = torch.zeros(batch_size, max_input_len)
        for b, l1 in enumerate(l_mask):
            attention_mask[b][:l1] = 1

        attention_mask = attention_mask.type_as(l_toks)
        l_mask = l_mask.type_as(l_toks)
        return attention_mask, l_mask

    @classmethod
    def save_analysis_results(
        cls,
        path_analysis_dir,
        test_type,
        test_score_dict,
        parse_stat,
        card_stat,
        corrects_parse,
        parses,
        pr_parses,
    ):
        os.makedirs(path_analysis_dir, exist_ok=True)

        gu.write_json(
            path_analysis_dir / f"{test_type}_score_dict.jsonl", test_score_dict
        )
        gu.write_json(path_analysis_dir / f"{test_type}_parse_stat.json", parse_stat)
        gu.write_json(path_analysis_dir / f"{test_type}_card_stat.json", card_stat)
        gu.write_jsonl(
            path_analysis_dir / f"{test_type}_correct_parse.jsonl",
            corrects_parse,
        )

        gu.write_jsonl(path_analysis_dir / f"{test_type}_parses.jsonl", parses)
        gu.write_jsonl(path_analysis_dir / f"{test_type}_pr_parses.jsonl", pr_parses)

    @classmethod
    def print_parsing_result(cls, parses, pr_parses, target_id=0):
        print("*" * 40)
        print("GT: ", parses[target_id])
        print("*" * 40)
        print("PR: ", pr_parses[target_id])
        print("*" * 40)

    @classmethod
    def get_encoded1_part(
        cls,
        all_encoder_layer,
        max_input_len,
        input_split_overlap_len,
        n_seps,
        i_sep,
        i_toks,
        j_toks,
        l_toks,
        encoder_layer_ids_used_in_decoder,
    ):
        encoded1 = collect_output_of_target_layer(
            all_encoder_layer, encoder_layer_ids_used_in_decoder
        )  # [batch, max_input_len, dim]

        encoded1_part = []
        for b, encoded11 in enumerate(encoded1):
            split_offset = i_toks[b][i_sep]
            overlap_offset = 0 if i_sep == 0 else input_split_overlap_len
            if j_toks[b][i_sep] != 0:  # it is zero only when already finished.
                if i_sep < n_seps[b] - 1:
                    st = i_toks[b][i_sep] - split_offset + overlap_offset
                    ed = j_toks[b][i_sep] - split_offset
                else:
                    # final one will take full tokens with more overlap.
                    if n_seps[b] == 1:
                        # final, yet only one
                        st = 0
                        ed = l_toks[b]
                    else:
                        # final of multiple splits
                        previous_ed = j_toks[b][i_sep - 1]
                        n_remaining_tok = l_toks[b] - previous_ed
                        st = max_input_len - n_remaining_tok
                        ed = max_input_len
            else:
                st = 0
                ed = 0

            encoded1_part.append(encoded1[b][st:ed])
        return encoded1_part

    @classmethod
    def tensorize_encoded(cls, encoded, l_toks, lmax_toks):
        """
        make tensor and pad 0
        encoded = [split, batch, input_len, dim]
        """
        # lmax_toks = max(l_toks)
        new_encoded = []
        n_sep = len(encoded)
        batch_size = len(encoded[0])
        for b in range(batch_size):
            new_encoded1 = []
            for i_sep in range(n_sep):
                new_encoded1.append(encoded[i_sep][b])
            new_encoded1 = torch.cat(new_encoded1, dim=0)  # input_len dimension,
            # [b, input_len, dim]
            n_pad = lmax_toks - l_toks[b]
            new_encoded.append(F.pad(new_encoded1, [0, 0, 0, n_pad]).unsqueeze(0))

        new_encoded = torch.cat(new_encoded, dim=0)
        return new_encoded

    @classmethod
    def get_split_param(
        cls,
        text_toks,
        max_input_len,
        input_split_overlap_len,
        type_informer_tensor,
        do_split=True,
    ):
        n_seps = []
        i_toks = []
        j_toks = []
        l_toks = []
        for text_tok in text_toks:
            n_sep, i_tok, j_tok, l_tok = get_split_param1(
                text_tok, max_input_len, input_split_overlap_len, do_split=do_split
            )
            n_seps.append(n_sep)
            i_toks.append(i_tok)
            j_toks.append(j_tok)
            l_toks.append(l_tok)

        assert type_informer_tensor.dtype == torch.long
        n_seps = torch.tensor(n_seps).type_as(type_informer_tensor)
        i_toks = pad_ids(
            i_toks, max_length=max(n_seps), type_informer_tensor=type_informer_tensor
        )
        j_toks = pad_ids(
            j_toks, max_length=max(n_seps), type_informer_tensor=type_informer_tensor
        )
        l_toks = torch.tensor(l_toks).type_as(type_informer_tensor)

        return n_seps, i_toks, j_toks, l_toks

    @classmethod
    def get_score(cls, model, batch_data_in, lmax_boxes):
        (
            text_tok_ids,
            rn_center_toks,
            rn_dist_toks,
            rn_angle_toks,
            vertical_toks,
            char_size_toks,
            header_toks,
            n_seps,
            i_toks,
            j_toks,
            l_toks,
        ) = batch_data_in

        rn_center_x_toks, rn_center_y_toks = separate_center_coord_to_xy(rn_center_toks)

        (
            text_tok_ids,
            rn_center_x_toks,
            rn_center_y_toks,
            rn_dist_toks,
            rn_angle_toks,
            vertical_toks,
            char_size_toks,
            header_toks,
        ) = convert_feature_to_tensor(
            max(l_toks),
            text_tok_ids,
            rn_center_x_toks,
            rn_center_y_toks,
            rn_dist_toks,
            rn_angle_toks,
            vertical_toks,
            char_size_toks,
            header_toks,
        )

        n_seps, i_toks, j_toks, l_toks = convert_split_params_to_tensor(
            n_seps, i_toks, j_toks, l_toks
        )
        lmax_toks = l_toks.max().item()
        _b = len(n_seps)
        score = model(
            text_tok_ids,
            rn_center_x_toks,
            rn_center_y_toks,
            rn_dist_toks,
            rn_angle_toks,
            vertical_toks,
            char_size_toks,
            header_toks,
            n_seps,
            i_toks,
            j_toks,
            l_toks,
            lmax_toks,  # for multi-gpu cases
            lmax_boxes,
        )

        return score

    @classmethod
    def generate_score_dict(
        cls, mode, avg_loss, p_edge, r_edge, f1_edge, f1, is_tensor=True
    ):
        if is_tensor:
            score_dict = {
                f"{mode}__avg_loss": avg_loss.item(),
                f"{mode}__f1": f1.item(),
                f"{mode}__precision_edge_avg": torch.mean(p_edge).item(),
                f"{mode}__recall_edge_avg": torch.mean(r_edge).item(),
                f"{mode}__f1_edge_avg": torch.mean(f1_edge).item(),
            }

            for i, (p_edge1, r_edge1, f1_edge1) in enumerate(
                zip(p_edge, r_edge, f1_edge)
            ):
                sub_score_dict = {
                    f"{mode}__precision_edge_of_type_{i}": p_edge1.item(),
                    f"{mode}__recall_edge_of_type_{i}": r_edge1.item(),
                    f"{mode}__f1_edge_of_type_{i}": f1_edge1.item(),
                }
                score_dict.update(sub_score_dict)
        else:
            score_dict = {
                f"{mode}__avg_loss": avg_loss.item(),
                f"{mode}__f1": f1,
                f"{mode}__precision_edge_avg": np.mean(p_edge),
                f"{mode}__recall_edge_avg": np.mean(r_edge),
                f"{mode}__f1_edge_avg": np.mean(f1_edge),
            }

            for i, (p_edge1, r_edge1, f1_edge1) in enumerate(
                zip(p_edge, r_edge, f1_edge)
            ):
                sub_score_dict = {
                    f"{mode}__precision_edge_of_type_{i}": p_edge1,
                    f"{mode}__recall_edge_of_type_{i}": r_edge1,
                    f"{mode}__f1_edge_of_type_{i}": f1_edge1,
                }
                score_dict.update(sub_score_dict)

        return score_dict

    @classmethod
    def collect_outputs_gt(cls, task, outputs, return_aux=False):
        label_units = cls.gather_values_from_step_outputs(outputs, "label_units")
        parses = cls.gather_values_from_step_outputs(outputs, "parses")
        if task == "funsd" and return_aux:
            text_unit_field_labels = cls.gather_values_from_step_outputs(
                outputs, "text_unit_field_labels"
            )
            f_parse_box_ids = cls.gather_values_from_step_outputs(
                outputs, "f_parse_box_ids"
            )

            return (label_units, parses, text_unit_field_labels, f_parse_box_ids)
        else:
            return (label_units, parses)

    @classmethod
    def collect_outputs_pr(cls, task, outputs, return_aux=False):
        pr_label_units = cls.gather_values_from_step_outputs(outputs, "pr_label_units")
        pr_parses = cls.gather_values_from_step_outputs(outputs, "pr_parses")
        if task == "funsd" and return_aux:
            pr_text_unit_field_labels = cls.gather_values_from_step_outputs(
                outputs, "pr_text_unit_field_labels"
            )
            pr_f_parse_box_ids = cls.gather_values_from_step_outputs(
                outputs, "pr_f_parse_box_ids"
            )
            return (
                pr_label_units,
                pr_parses,
                pr_text_unit_field_labels,
                pr_f_parse_box_ids,
            )
        else:
            return (pr_label_units, pr_parses)

    @classmethod
    def gather_values_from_step_outputs(cls, outputs, key):
        # each_step/batch
        # [combine each batch resuilts]
        return reduce(
            lambda out_accum, results_each_step: out_accum + results_each_step,
            [x[key] for x in outputs],
            [],
        )

    @classmethod
    def cal_f1_scores(cls, task, mode, parses1, parses2, parse_refine_options):
        if parses1[0] is None:
            f1 = 0
            parse_stat = {"total": [0, 0, 0, 0, 0, 0]}
            card_stat = None
            corrects_parse = None

        else:
            if task == "funsd":
                # print("The calculation of FUNSD parse-F1 is not supported.")
                f1 = -1
                parse_stat = {"total": [-1] * 6}
                card_stat = {}
                corrects_parse = []
            elif task == "receipt_v1":
                f1, parse_stat, card_stat, corrects_parse = au.cal_parsing_score(
                    mode,
                    parses1,
                    parses2,
                    task,
                    reformat=(True, True),
                    refine_parse=parse_refine_options["refine_parse"],
                    allow_small_edit_distance=parse_refine_options[
                        "allow_small_edit_distance"
                    ],
                    task_lan=parse_refine_options["task_lan"],
                    unwanted_fields=parse_refine_options["unwanted_fields"],
                )
            else:
                raise NotImplementedError

        return f1, parse_stat, card_stat, corrects_parse

    @classmethod
    def count_tp_fn_fp(
        cls,
        task,
        label_units,
        pr_label_units,
        mode,
        parses,
        pr_parses,
        parse_refine_options,
    ):

        tp_edge, fn_edge, fp_edge = au.cal_tp_fn_fp_of_edges(
            label_units, pr_label_units
        )

        _, parse_stat, _, _ = cls.cal_f1_scores(
            task, mode, parses, pr_parses, parse_refine_options
        )

        # [TP, FP, FN, Precision, Recall, F1]
        tp_parse, fp_parse, fn_parse, _, _, _ = parse_stat["total"]

        return tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse
