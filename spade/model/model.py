# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import os
import time
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from pprint import pprint
from typing import Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

import spade.model.model_utils as mu
import spade.utils.analysis_utils as au
import spade.utils.general_utils as gu
from spade.model.metric import SpadeMetric
from spade.model.model_loss import Loss_rt
from spade.model.model_optimizer import get_lr_dict, get_optimizer
from spade.model.model_spade_encoder import SpadeEncoder
from spade.model.model_spade_graph_decoder import gen_parses, pred_label
from spade.model.model_spade_graph_generator import SpadeDecoder
from spade.model.model_utils import RelationTaggerUtils as rtu


class RelationTagger(pl.LightningModule):
    # No pool layer
    def __init__(self, hparam, tparam, iparam, path_data_folder, verbose=False):
        """ """
        super().__init__()

        self.hparam = hparam
        self.tparam = tparam
        self.iparam = iparam
        self.verbose = verbose

        self.task = hparam.task
        self.task_lan = hparam.task_lan
        self.fields = hparam.fields
        self.field_rs = hparam.field_representers
        self.n_fields = len(hparam.fields)
        self.name = hparam.model_name
        self.max_input_len = hparam.max_input_len
        self.input_split_overlap_len = hparam.input_split_overlap_len
        self.encoder_layer_ids_used_in_decoder = (
            hparam.encoder_layer_ids_used_in_decoder
        )
        self.cross_entropy_loss_weight = torch.tensor(tparam.cross_entropy_loss_weight)
        self.validation_top_score = None  # the higher, the better

        self.encoder_layer = gen_encoder_layer(hparam, path_data_folder)  # encoder

        self.decoder_layer = gen_decoder_layer(
            hparam, self.encoder_layer.transformer_cfg
        )

        self.spade_metric = SpadeMetric(hparam.n_relation_type, dist_sync_on_step=False)
        self.parse_refine_options = {
            "refine_parse": self.iparam.refine_parse,
            "allow_small_edit_distance": self.iparam.allow_small_edit_distance,
            "task_lan": self.task_lan,
            "unwanted_fields": self.iparam.unwanted_fields,
        }

        self.char_for_detokenization = gu.get_char_for_detokenization(
            hparam.encoder_backbone_name
        )

    def forward(
        self,
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
        lmax_toks,  # for multi-gpu
        lmax_boxes,
    ):
        # splits
        header_ori_toks = deepcopy(header_toks)
        encoded = self.encoder_forward(
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
            lmax_toks,
        )
        # decoding
        score = self.decoder_layer(encoded, header_ori_toks, lmax_boxes)
        return score

    def encoder_forward(
        self,
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
        lmax_toks,
    ):

        # 1. split features that have len > 512
        (
            text_tok_ids,
            rn_center_x_toks,
            rn_center_y_toks,
            rn_dist_toks,
            rn_angle_toks,
            vertical_toks,
            char_size_toks,
            header_toks,
        ) = rtu.split_features(
            n_seps,
            i_toks,
            j_toks,
            self.max_input_len,
            text_tok_ids,
            rn_center_x_toks,
            rn_center_y_toks,
            rn_dist_toks,
            rn_angle_toks,
            vertical_toks,
            char_size_toks,
            header_toks,
        )  # [n_sep, batch_size]

        # 2. encode each splitted feature
        encoded = []
        nmax_seps = max(n_seps)
        for i_sep in range(nmax_seps):
            attention_mask, l_mask = rtu.gen_input_mask(
                i_sep, l_toks, i_toks, j_toks, self.max_input_len
            )
            try:
                all_encoder_layer = self.encoder_layer(
                    text_tok_ids[i_sep],
                    rn_center_x_toks[i_sep],
                    rn_center_y_toks[i_sep],
                    rn_dist_toks[i_sep],
                    rn_angle_toks[i_sep],
                    vertical_toks[i_sep],
                    char_size_toks[i_sep],
                    header_toks[i_sep],
                    attention_mask=attention_mask,
                )

            except RuntimeError:
                print(f"i_sep = {i_sep + 1} / n_sep = {nmax_seps}")
                print("Fail to encode due to the memory limit.")
                print("The encoder output vectors set to zero.")
                if i_sep == 0:
                    print(f"Even single encoding faield!")
                    raise MemoryError
                else:
                    l_layer = self.encoder_layer.transformer_cfg.num_hidden_layers

                    all_encoder_layer = [
                        torch.zeros_like(all_encoder_layer[0]) for _ in range(l_layer)
                    ]

            encoded1_part = rtu.get_encoded1_part(
                all_encoder_layer,
                self.max_input_len,
                self.input_split_overlap_len,
                n_seps,
                i_sep,
                i_toks,
                j_toks,
                l_toks,
                self.encoder_layer_ids_used_in_decoder,
            )

            encoded.append(encoded1_part)

        # 3. Combine splited encoder outputs
        encoded = rtu.tensorize_encoded(encoded, l_toks, lmax_toks)
        return encoded

    # @gu.timeit
    def _run(self, mode, batch):

        # 1. Batchwise collection of features
        (
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
        ) = mu.collect_features_batchwise(batch)

        # 2. Calculate length for the padding.
        l_boxes = [len(x) for x in texts]
        l_tokens = [len(x) for x in text_toks]

        if self.hparam.token_lv_boxing:
            # Individual units are tokens.
            l_units = l_tokens
            label_units = label_toks
            text_units = text_toks
        else:
            # Individual units are text segments from OCR-detection-boxes.
            l_units = l_boxes
            label_units = labels
            text_units = texts

        # hot-fix
        # label_toks include junks when label is None.
        if labels[0] is None:
            label_units = labels

        lmax_boxes = max(l_boxes)

        # 3. Split data whose token length > 512
        n_seps, i_toks, j_toks, l_toks = rtu.get_split_param(
            text_toks,
            self.hparam.max_input_len,
            self.hparam.input_split_overlap_len,
            type_informer_tensor=text_tok_ids[0],
        )

        # 4. get score
        batch_data_in = (
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
        )

        score = rtu.get_score(self, batch_data_in, lmax_boxes)

        # 5. prediction
        pr_label_units = pred_label(
            self.task,
            score,
            self.hparam.inferring_method,
            self.n_fields,
            l_units,
        )
        if labels[0] is not None:
            # 6. Generate gt parse
            parses, f_parses, text_unit_field_labels, f_parse_box_ids = gen_parses(
                self.task,
                self.fields,
                self.field_rs,
                text_units,
                label_units,
                header_toks,
                l_max_gen=self.hparam.l_max_gen_of_each_parse,
                max_info_depth=self.hparam.max_info_depth,
                strict=True,
                token_lv_boxing=self.hparam.token_lv_boxing,
                backbone_name=self.hparam.encoder_backbone_name,
            )
        else:
            parses = [None] * len(texts)
            f_parses = [None] * len(texts)
            text_unit_field_labels = [None] * len(texts)
            f_parse_box_ids = [None] * len(texts)

        # for the speed, set max serialization length small at initial stage.
        if mode == "train" and self.current_epoch <= 100:
            pr_l_max_gen = 2
        else:
            pr_l_max_gen = self.hparam.l_max_gen_of_each_parse

        # 7. Generate predicted parses
        (
            pr_parses,
            pr_f_parses,
            pr_text_unit_field_labels,
            pr_f_parse_box_ids,
        ) = gen_parses(
            self.task,
            self.fields,
            self.field_rs,
            text_units,
            pr_label_units,
            header_toks,
            l_max_gen=pr_l_max_gen,
            max_info_depth=self.hparam.max_info_depth,
            strict=False,
            token_lv_boxing=self.hparam.token_lv_boxing,
            backbone_name=self.hparam.encoder_backbone_name,
        )

        results = {
            "data_ids": data_ids,
            "score": score,
            "text_units": text_units,
            "label_units": label_units,
            "pr_label_units": pr_label_units,
            "l_units": l_units,
            "parses": parses,
            "pr_parses": pr_parses,
            "text_unit_field_labels": text_unit_field_labels,
            "pr_text_unit_field_labels": pr_text_unit_field_labels,
            "f_parse_box_ids": f_parse_box_ids,
            "pr_f_parse_box_ids": pr_f_parse_box_ids,
        }

        return results

    def training_step(self, batch, batch_idx):
        results = self._run("train", batch)
        loss = Loss_rt(
            results["score"],
            results["label_units"],
            self.n_fields,
            results["l_units"],
            self.cross_entropy_loss_weight,
        )
        self.log("training_loss", loss, sync_dist=True)
        out = {"loss": loss}

        if gu.get_local_rank() == 0:
            training_out = {
                "training_loss": loss,
                "data_ids": results["data_ids"],
                "parses": results["parses"],
                "pr_parses": results["pr_parses"],
                "label_units": results["label_units"],
                "pr_label_units": results["pr_label_units"],
            }
            out.update({"training_out": training_out})
        return out

    def training_epoch_end(self, outputs) -> None:
        if gu.get_local_rank() == 0:
            losses = [x["loss"] for x in outputs]
            avg_loss = torch.mean(torch.stack(losses))
            print(f"Training, ave_loss = {avg_loss}")

            print(f"Training, gt_parse e.g. {outputs[0]['training_out']['parses'][0]}")
            print(
                f"Training, pr_parse e.g. {outputs[0]['training_out']['pr_parses'][0]}"
            )
            print(
                f"Epoch {self.current_epoch}, average training loss: {avg_loss.item()}"
            )

    def validation_step(self, batch, batch_idx):
        results = self._run("test", batch)
        loss = Loss_rt(
            results["score"],
            results["label_units"],
            self.n_fields,
            results["l_units"],
            self.cross_entropy_loss_weight,
        )

        self.log("validation_loss", loss, sync_dist=True)
        val_out = {
            "data_ids": results["data_ids"],
            "loss": loss,
            "label_units": results["label_units"],
            "pr_label_units": results["pr_label_units"],
            "parses": results["parses"],
            "pr_parses": results["pr_parses"],
        }

        (tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse) = rtu.count_tp_fn_fp(
            self.task,
            results["label_units"],
            results["pr_label_units"],
            "validation",
            results["parses"],
            results["pr_parses"],
            self.parse_refine_options,
        )

        self.spade_metric.update(
            tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse
        )

        return val_out

    def validation_epoch_end(self, outputs):
        # 1. Reduce validation results
        (
            precision_edge_avg,
            recall_edge_avg,
            f1_edge_avg,
            precision_parse,
            recall_parse,
            f1_parse,
        ) = self.spade_metric.compute()
        self.spade_metric.reset()

        if gu.get_local_rank() == 0:
            (label_units, parses) = rtu.collect_outputs_gt(self.task, outputs)
            (_, pr_parses) = rtu.collect_outputs_pr(self.task, outputs)

            # 2. Compute validation score
            losses = [x["loss"] for x in outputs]
            avg_loss = torch.mean(torch.stack(losses))

            validation_score_dict = rtu.generate_score_dict(
                "validation",
                avg_loss,
                precision_edge_avg,
                recall_edge_avg,
                f1_edge_avg,
                f1_parse,
            )

            validation_score = validation_score_dict[
                f"validation__{self.tparam.validation_metric}"
            ]

            print(f"Val. metric: {self.tparam.validation_metric}")
            print(
                f"Val. top score: {self.validation_top_score}, current val. score: {validation_score}"
            )
            print(f"Epoch: {self.current_epoch}, validation score_dict")
            pprint(f"{validation_score_dict}")

            # 3. Update best validation score
            if self.validation_top_score is None:
                self.validation_top_score = validation_score

            if self.validation_top_score < validation_score:
                print(
                    f"Best score = {validation_score} from epoch {self.current_epoch}"
                )
                self.validation_top_score = validation_score
                path_save_model = Path(self.tparam.path_save_model_dir) / "best"
                os.makedirs(path_save_model, exist_ok=True)
                gu.save_pytorch_model(path_save_model, self)

            if (
                self.current_epoch % self.tparam.save_epoch_interval == 0
                and self.current_epoch > 0
            ):
                path_save_model = (
                    Path(self.tparam.path_save_model_dir) / self.current_epoch
                )
                os.makedirs(path_save_model, exist_ok=True)
                gu.save_pytorch_model(path_save_model, self)

            if self.verbose:
                if self.current_epoch != 0:
                    print(f"Validation result at epoch {self.current_epoch}")
                    print(f"{validation_score_dict}")
                    rtu.print_parsing_result(parses, pr_parses)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.tparam, self)
        lr_dict = get_lr_dict(optimizer, self.tparam)

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def test_step(self, batch, batch_idx, dataset_idx):
        results = self._run("test", batch)
        if results["label_units"][0] is None:  # parse ocr results
            loss = -1
        else:
            loss = Loss_rt(
                results["score"],
                results["label_units"],
                self.n_fields,
                results["l_units"],
                self.cross_entropy_loss_weight,
            )

        test_out = {
            "dataset_idx": dataset_idx,
            "loss": loss,
            "data_ids": results["data_ids"],
            "label_units": results["label_units"],
            "pr_label_units": results["pr_label_units"],
            "parses": results["parses"],
            "pr_parses": results["pr_parses"],
            "text_unit_field_labels": results["text_unit_field_labels"],
            "pr_text_unit_field_labels": results["pr_text_unit_field_labels"],
            "f_parse_box_ids": results["f_parse_box_ids"],
            "pr_f_parse_box_ids": results["pr_f_parse_box_ids"],
        }

        (tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse) = rtu.count_tp_fn_fp(
            self.task,
            results["label_units"],
            results["pr_label_units"],
            "test",
            results["parses"],
            results["pr_parses"],
            self.parse_refine_options,
        )
        self.spade_metric.update(
            tp_edge, fp_edge, fn_edge, tp_parse, fp_parse, fn_parse
        )

        return test_out

    @rank_zero_only
    def test_epoch_end(self, outputs: List[Any]) -> None:
        if self.hparam.task == "receipt_v1":
            self.test_epoch_end_receipt_v1(outputs)
        elif self.hparam.task == "funsd":
            self.test_epoch_end_funsd(outputs)
        else:
            raise NotImplementedError

    @rank_zero_only
    @gu.timeit
    def predict_step(self, batch, batch_idx, dataset_idx):
        results = self._run("test", batch)
        assert len(results["data_ids"]) == 1
        test_out = {
            "data_id": results["data_ids"][0],
            "text_unit": results["text_units"][0],
            "pr_parse": results["pr_parses"][0],
            "pr_label": results["pr_label_units"][0],
            "pr_text_unit_field_label": results["pr_text_unit_field_labels"][0],
        }
        return test_out

    def test_epoch_end_funsd(self, outputs):
        test_types = ["dev", "test"]

        for i, (test_type, outputs_each_dataloader) in enumerate(
            zip_longest(test_types, outputs)
        ):
            if i == 0:
                continue  # dev == test in funsd case.

            # 1. Collect required features from the outputs
            losses = [x["loss"] for x in outputs_each_dataloader]
            avg_loss = torch.mean(torch.stack(losses))

            (
                label_units,
                parses,
                text_unit_field_labels,
                f_parse_box_ids,
            ) = rtu.collect_outputs_gt(
                self.task, outputs_each_dataloader, return_aux=True
            )
            label_units = [x.cpu().numpy() for x in label_units]
            (
                pr_label_units,
                pr_parses,
                pr_text_unit_field_labels,
                pr_f_parse_box_ids,
            ) = rtu.collect_outputs_pr(
                self.task, outputs_each_dataloader, return_aux=True
            )

            # 2. Calculate score
            # 2.1 Calculate without the field header indicators.
            tp_edge, fn_edge, fp_edge = au.cal_tp_fn_fp_of_edges(
                label_units, pr_label_units
            )

            p_edge, r_edge, f1_edge = au.cal_p_r_f1(tp_edge, fn_edge, fp_edge)

            f1, parse_stat, card_stat, corrects_parse = rtu.cal_f1_scores(
                self.task, "test", parses, pr_parses, self.parse_refine_options
            )

            fields_of_interest = [
                "header.header",
                "qa.question",
                "qa.answer",
                "other.other",
            ]

            # 2.2 Calculate with the field header indicators.
            (
                header_id_of_entities,
                header_label_of_entities,
            ) = au.extract_header_id_of_entities(f_parse_box_ids)

            text_unit_field_label_subs = au.get_headers_of(
                header_id_of_entities, text_unit_field_labels
            )
            pr_text_unit_field_label_subs = au.get_headers_of(
                header_id_of_entities, pr_text_unit_field_labels
            )

            tp_fn_fp_all_entity = au.get_tp_fn_fp_all(
                text_unit_field_label_subs,
                pr_text_unit_field_label_subs,
                fields_of_interest,
            )
            # 2.2.1 Calculate F1 for ELB task.
            p_r_f1_entity, p_r_f1_all_entity = au.get_p_r_f1_entity(
                tp_fn_fp_all_entity, fields_of_interest
            )

            # 2.2.2 Calculate F1 for ELK task.
            gt_links = au.extract_links(self.fields, label_units, target_label_id=1)
            pr_links = au.extract_links(self.fields, pr_label_units, target_label_id=1)

            pr_links_filtered = au.filter_non_header_id(
                pr_links, header_id_of_entities, gt=False
            )
            p_r_f1_link = au.get_p_r_f1_link(gt_links, pr_links_filtered)

            test_score_dict = rtu.generate_score_dict(
                test_type, avg_loss, p_edge, r_edge, f1_edge, f1, is_tensor=False
            )
            test_score_dict.update(
                {
                    "p_r_f1_entity": p_r_f1_entity,
                    "p_r_f1_all_entity_ELB": p_r_f1_all_entity,  # ELB
                    "p_r_f1_link_ELK": p_r_f1_link,  # ELK
                }
            )

            # 3. Save analysis results
            path_analysis_dir = Path(self.hparam.path_analysis_dir)
            rtu.save_analysis_results(
                path_analysis_dir,
                test_type,
                test_score_dict,
                parse_stat,
                card_stat,
                corrects_parse,
                parses,
                pr_parses,
            )

    def test_epoch_end_receipt_v1(self, outputs):
        test_types = ["dev", "test", "ocr_dev", "ocr_test"]

        parsess = []
        data_idss = []
        ave_losses = []

        for i, (test_type, outputs_each_dataloader) in enumerate(
            zip_longest(test_types, outputs)
        ):
            # 1. Collect required results from the outputs
            data_ids = rtu.gather_values_from_step_outputs(
                outputs_each_dataloader, "data_ids"
            )

            (pr_label_units, pr_parses) = rtu.collect_outputs_pr(
                self.task, outputs_each_dataloader
            )

            if i <= 1:
                # dev, test
                losses = [x["loss"] for x in outputs_each_dataloader]
                avg_loss = torch.mean(torch.stack(losses))
                ave_losses.append(avg_loss)

                (label_units, parses) = rtu.collect_outputs_gt(
                    self.task, outputs_each_dataloader
                )
                parsess.append(parses)
                data_idss.append(data_ids)

                tp_edge, fn_edge, fp_edge = au.cal_tp_fn_fp_of_edges(
                    label_units, pr_label_units
                )
                p_edge, r_edge, f1_edge = au.cal_p_r_f1(tp_edge, fn_edge, fp_edge)

            else:
                # ocr-dev, ocr-test
                # ocr-dev, ocr-test do not include ground-truth.
                # Thus, ground-truth from dev, test are used for the calculation.
                data_ids = [
                    int(x) for x in data_ids
                ]  # ocr-dev, -test data use string data_id.
                avg_loss = ave_losses[i % 2]
                parses_unsorted = deepcopy(parsess[i % 2])
                data_ids_oracle = deepcopy(data_idss[i % 2])

                # sort parses
                _ii = [data_ids_oracle.index(id) for id in data_ids]
                parses = [parses_unsorted[i] for i in _ii]

                # Cannot calculate dependency parsing scores in ocr-dev, ocr-test
                p_edge, r_edge, f1_edge = [-1], [-1], [-1]

            # 2. Calculate scores
            f1, parse_stat, card_stat, corrects_parse = rtu.cal_f1_scores(
                self.task,
                "test",
                parses,
                pr_parses,
                self.parse_refine_options,
            )
            test_score_dict = rtu.generate_score_dict(
                test_type, avg_loss, p_edge, r_edge, f1_edge, f1, is_tensor=False
            )

            # 3. Save analysis results
            path_analysis_dir = Path(self.hparam.path_analysis_dir)
            rtu.save_analysis_results(
                path_analysis_dir,
                test_type,
                test_score_dict,
                parse_stat,
                card_stat,
                corrects_parse,
                parses,
                pr_parses,
            )


def gen_encoder_layer(hparam, path_data_folder):
    # 1. Load pretrained transformer
    if hparam.encoder_backbone_is_pretrained:
        (
            pretrained_transformer,
            pretrained_transformer_cfg,
        ) = mu.get_pretrained_transformer(
            path_data_folder,
            hparam.encoder_backbone_name,
            hparam.encoder_backbone_tweak_tag,
        )

    # 2. Load model
    if hparam.encoder_type_name == "spade":
        # 2.1 Load encoder
        spatial_text_encoder = SpadeEncoder(hparam, path_data_folder)

        # 2.2 Initialize the subset of weights from the pretraiend transformer.
        if hparam.encoder_backbone_is_pretrained:
            print(f"pretrained {hparam.encoder_backbone_name} is used")
            mu.check_consistency_between_backbone_and_encoder(
                pretrained_transformer_cfg, spatial_text_encoder.transformer_cfg
            )
            pretrained_transformer_state_dict = pretrained_transformer.state_dict()
            spatial_text_encoder = gu.update_part_of_model(
                parent_model_state_dict=pretrained_transformer_state_dict,
                child_model=spatial_text_encoder,
                rank=gu.get_local_rank(),
            )
    else:
        raise NotImplementedError

    return spatial_text_encoder


def gen_decoder_layer(hparam, encoder_transformer_cfg):
    input_size = encoder_transformer_cfg.hidden_size * len(
        hparam.encoder_layer_ids_used_in_decoder
    )
    # 1. Load decoder
    if hparam.decoder_type == "spade":
        decoder_layer = SpadeDecoder(
            input_size,
            hparam.decoder_hidden_size,
            hparam.n_relation_type,
            hparam.fields,
            hparam.token_lv_boxing,
            hparam.include_second_order_relations,
            hparam.vi_params,
        )
    else:
        raise NotImplementedError
    return decoder_layer
