# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import spade.model.data_utils as du
import spade.model.model_utils as mu
import spade.utils.general_utils as gu


class SpadeDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.task = cfg.model_param.task
        self.path_data_folder = Path(cfg.path_data_folder)
        self.data_paths = cfg.data_paths
        self.batch_size = cfg.train_param.batch_size
        self.batch_size_for_test = cfg.train_param.batch_size_for_test
        self.cfg = cfg
        self.tokenizer = mu.get_tokenizer(
            cfg.path_data_folder,
            cfg.model_param.encoder_backbone_name,
            cfg.model_param.encoder_backbone_tweak_tag,
        )

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            self.train_data, self.dev_data = self._prepare_train_data()
        if stage == "test":
            self.test_datas = self._prepare_test_datas()
            # prepare test

    def _prepare_train_data(self):
        train_data = get_data(
            self.path_data_folder / self.data_paths["train"],
            "train",
            self.cfg,
            self.tokenizer,
        )
        dev_data = get_data(
            self.path_data_folder / self.data_paths["dev"],
            "test",
            self.cfg,
            self.tokenizer,
        )

        return train_data, dev_data

    def _prepare_test_datas(self):
        if self.task in ["receipt_v1"]:
            return self._prepare_test_datas_include_ocr_input()
        elif self.task in ["funsd"]:
            return self._prepare_test_datas_no_ocr_input()
        else:
            raise NotImplementedError

    def _prepare_predict_data(self, path_predict_input_json):
        predict_data = get_data(
            path_predict_input_json, "infer", self.cfg, self.tokenizer, is_json=True
        )
        return predict_data

    def _prepare_test_datas_no_ocr_input(self):
        dev_data = get_data(
            self.path_data_folder / self.data_paths["dev"],
            "test",
            self.cfg,
            self.tokenizer,
        )
        test_data = get_data(
            self.path_data_folder / self.data_paths["test"],
            "test",
            self.cfg,
            self.tokenizer,
        )

        return dev_data, test_data

    def _prepare_test_datas_include_ocr_input(self):

        dev_data = get_data(
            self.path_data_folder / self.data_paths["dev"],
            "test",
            self.cfg,
            self.tokenizer,
        )
        test_data = get_data(
            self.path_data_folder / self.data_paths["test"],
            "test",
            self.cfg,
            self.tokenizer,
        )
        # op: ocr parsing
        op_dev_data = get_data(
            self.path_data_folder / self.data_paths["op_dev"],
            "infer",
            self.cfg,
            self.tokenizer,
        )
        op_test_data = get_data(
            self.path_data_folder / self.data_paths["op_test"],
            "infer",
            self.cfg,
            self.tokenizer,
        )

        return dev_data, test_data, op_dev_data, op_test_data

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            batch_size=self.batch_size,
            dataset=self.train_data,
            shuffle=True,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        return loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=self.dev_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        return loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.task in ["receipt_v1"]:
            return self._test_dataloader_including_ocr_input()
        elif self.task in ["funsd"]:
            return self._test_dataloader()
        else:
            raise NotImplementedError

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        predict_data = self._prepare_predict_data(self.path_predict_input_json)
        loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=predict_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        return loader

    def _test_dataloader(self):
        dev_data, test_data = self.test_datas
        dev_loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=dev_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        test_loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=test_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )

        return [dev_loader, test_loader]

    def _test_dataloader_including_ocr_input(self):
        dev_data, test_data, op_dev_data, op_test_data = self.test_datas
        dev_loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=dev_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )
        test_loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=test_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )

        op_dev_loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=op_dev_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )

        op_test_loader = DataLoader(
            batch_size=self.batch_size_for_test,
            dataset=op_test_data,
            shuffle=False,
            num_workers=self.cfg.train_param.n_cpus,
            collate_fn=lambda x: x,
        )

        return [dev_loader, test_loader, op_dev_loader, op_test_loader]


def get_data(fpath, mode, cfg, tokenizer, is_json=False):
    if is_json:
        raw_data = [gu.load_json(fpath)]
    else:
        raw_data = gu.load_jsonl(fpath, cfg["toy_data"], cfg["toy_size"])
    data = SpadeData(raw_data, mode, cfg, tokenizer, fpath)
    return data


class SpadeData(torch.utils.data.Dataset):
    def __init__(self, raw_data, mode, cfg, tokenizer, fpath):
        self.task = cfg.model_param.task
        self.mode = mode

        self.tokenizer = tokenizer
        self.fields = cfg.model_param.fields
        self.field_rs = cfg.model_param.field_representers
        self.n_fields = len(self.fields)

        self.method_for_token_xy_generation = cfg.method_for_token_xy_generation
        self.dist_norm = cfg.dist_norm
        self.n_dist_unit = cfg.model_param.n_dist_unit
        self.n_char_unit = cfg.model_param.n_char_unit
        self.n_angle_unit = cfg.model_param.n_angle_unit
        self.omit_angle_cal = cfg.model_param.omit_angle_cal

        self.augment_data = cfg.train_param.augment_data
        self.token_aug_param = cfg.train_param.initial_token_aug_params
        self.augment_coord = (cfg.train_param.augment_coord,)
        self.coord_aug_params = cfg.train_param.initial_coord_aug_params

        self.raw_data_input_type = cfg.raw_data_input_type

        self.data = self._normalize_raw_data(raw_data)

        self.token_pool = du.gen_token_pool(
            self.raw_data_input_type, self.tokenizer, self.data
        )
        self.fpath = fpath

    def gen_type1_data(self):
        def _normalize_label(label):
            if isinstance(label[0], np.ndarray):
                label = [x.tolist() for x in label]
            return label

        assert self.raw_data_input_type == "type0"
        # 1. Generate type1 data
        new_data = []
        for data1 in self.data:
            (
                data_id,
                text,
                coord,
                vertical,
                label,
                img_sz,
                img_feature,
                img_url,
            ) = self._get_each_field_from_raw_data(data1)
            label = _normalize_label(label)
            new_data1 = {
                "data_id": data_id,
                "fields": self.fields,
                "field_rs": self.field_rs,
                "text": text,
                "label": list(label) if label is not None else None,
                "coord": coord,
                "vertical": vertical,
                "img_sz": img_sz,
                "img_feature": img_feature,
                "img_url": img_url,
            }
            new_data.append(new_data1)

        # 2. save
        fpath_str = self.fpath.__str__()
        print(f"Working on {fpath_str}")
        assert fpath_str.endswith(".jsonl")
        if fpath_str.endswith("_type0.jsonl"):
            path_save = fpath_str[:-12] + "_type1.jsonl"
        else:
            path_save = fpath_str[:-6] + "_type1.jsonl"
        gu.write_jsonl(path_save, new_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            feature = self.gen_feature(
                self.data[idx],
                self.augment_data,
                self.token_aug_param,
                self.augment_coord,
                self.coord_aug_params,
            )
        else:
            feature = self.gen_feature(
                self.data[idx],
                augment_data=False,
                token_aug_params=None,
                augment_coord=False,
                coord_aug_params=None,
            )
        return feature

    def _normalize_raw_data(self, raw_data):
        if self.raw_data_input_type == "type0":
            data = self._normalize_raw_data_type0(raw_data)

        elif self.raw_data_input_type == "type1":
            # spade-optimized data format
            data = raw_data
        else:
            raise NotImplementedError

        return data

    def _normalize_raw_data_type0(self, raw_data):
        data = []
        for raw_data1 in raw_data:
            t1 = {}
            if self.mode != "infer":
                label, feature = du.get_label_and_feature(
                    raw_data1, self.task, self.fields, self.field_rs
                )

                img_sz, confidence, data_id = du.get_meta_feature(
                    self.task, raw_data1, feature
                )
            else:  # infer
                (
                    label,
                    feature,
                    confidence,
                    img_sz,
                    data_id,
                ) = du.get_label_and_feature_infer_mode(self.task, raw_data1)

            text = [x[0] for x in feature]
            coord = [x[1] for x in feature]
            is_vertical = [x[2] for x in feature]

            t1["data_id"] = data_id
            t1["label"] = label
            t1["ocr_feature"] = {
                "text": text,
                "coord": coord,
                "is_vertical": is_vertical,
                "confidence": confidence,
            }
            t1["img_sz"] = img_sz
            t1["img_feature"] = None  # shall be used later
            if "meta" in raw_data1:
                t1["image_url"] = raw_data1["meta"].get("image_url")
            else:
                t1["image_url"] = None
            data.append(t1)

        return data

    def gen_feature(
        self,
        data,
        augment_data,
        token_aug_params,
        augment_coord,
        coord_aug_params,
    ):
        if augment_data:
            assert self.mode == "train"

        (
            data_id,
            text,
            coord,
            vertical,
            label,
            img_sz,
            img_feature,
            image_url,
        ) = self._get_each_field_from_raw_data(data)

        if self.mode == "infer":
            (text, coord, vertical) = du.remove_blank_box(text, coord, vertical)

        if augment_coord:
            img = None
            clip_box_coord = True
            _, coord = du.gen_augmented_coord(
                img, coord, img_sz, coord_aug_params, clip_box_coord
            )

        (
            text_tok,
            coord_tok,
            direction_vec,
            direction_vec_tok,
            vertical_tok,
            char_size_tok,
            label_tok,
            header_tok,
        ) = self._tokenize_feature(
            text, coord, vertical, label, augment_data, token_aug_params
        )

        text_tok_id = self.tokenizer.convert_tokens_to_ids(text_tok)

        rel_center_tok, rel_dist_tok, rel_angle_tok = self.gen_rel_position(
            coord_tok, direction_vec_tok
        )
        rn_center_tok, rn_dist_tok, rn_angle_tok = self.normalize(
            rel_center_tok, rel_dist_tok, rel_angle_tok, img_sz, char_size_tok
        )

        char_size_tok = du.char_height_normalization(self.n_char_unit, char_size_tok)

        feature = {
            "data_id": data_id,
            "image_url": image_url,
            "text": text,
            "text_tok": text_tok,
            "text_tok_id": torch.as_tensor(text_tok_id),
            "label": torch.as_tensor(label) if self.mode != "infer" else label,
            "label_tok": torch.as_tensor(label_tok),
            "rn_center_tok": torch.as_tensor(rn_center_tok),
            "rn_dist_tok": torch.as_tensor(rn_dist_tok),
            "rn_angle_tok": torch.as_tensor(rn_angle_tok),
            "vertical_tok": torch.as_tensor(vertical_tok),
            "char_size_tok": torch.as_tensor(char_size_tok),
            "header_tok": torch.as_tensor(header_tok),
        }

        return feature

    def _get_each_field_from_raw_data(self, t1):
        if self.raw_data_input_type == "type1":
            data_id = t1["data_id"]
            text = t1["text"]
            coord = t1["coord"]
            vertical = t1["vertical"]
            label = t1["label"]
            img_sz = t1["img_sz"]
            img_feature = t1["img_feature"]
            image_url = t1["img_url"]
        elif self.raw_data_input_type == "type0":
            data_id = t1["data_id"]
            text = t1["ocr_feature"]["text"]
            coord = t1["ocr_feature"]["coord"]
            vertical = t1["ocr_feature"]["is_vertical"]
            label = t1["label"]
            if self.mode == "infer":
                assert label is None
            img_sz = t1["img_sz"]
            img_feature = t1["img_feature"]
            image_url = t1.get("image_url")
        else:
            raise NotImplementedError

        return data_id, text, coord, vertical, label, img_sz, img_feature, image_url

    def _tokenize_feature(
        self, text, coord, vertical, label, augment_data, token_aug_params
    ):
        if self.mode != "infer":
            pass
        else:
            assert label is None
            # generate junk label
            label = np.zeros([1, self.n_fields + len(text), len(text)])

        rel_idx = 1

        text_tok = []
        vertical_tok = []
        coord_tok = []
        char_size_tok = []
        direction_vec = []
        direction_vec_tok = []

        header_tok = np.ones(len(text), dtype=np.int)
        r_pnt = self.n_fields - 1
        c_pnt = -1

        label_sub = [np.array(label1, dtype=np.int) for label1 in label]
        for i, sub_features1 in enumerate(zip(text, coord, vertical)):
            text1, coord1, vertical1 = sub_features1
            text_tok1 = du.tokenizing_func(self.tokenizer, text1)
            if augment_data:
                text_tok1 = du.gen_augmented_text_tok1(
                    self.token_pool, text_tok1, token_aug_params
                )
            l_tok1 = len(text_tok1)
            coord1 = [np.array(xy) for xy in coord1]
            vertical_tok1 = du.augment_vertical(vertical1, l_tok1)

            csz = du.get_char_size1(coord1, vertical1)
            csz_tok1 = du.augment_char_size(csz, l_tok1)

            coord_tok1, direction_vec_tok1 = du.augment_coord(
                coord1,
                vertical1,
                l_tok1,
                self.method_for_token_xy_generation,
                text_tok1,
            )

            # text_tok.append(text_tok1)
            # coord_tok.append(coord_tok1)
            text_tok += text_tok1
            coord_tok += coord_tok1
            try:
                direction_vec.append(direction_vec_tok1[0])
            except:
                print(f"text1 = {text1}")
                print(f"text1 len = {len(text1)}")
                print(f"i={i}, text={text}, coord={coord}")
                print(text1, coord1, l_tok1, text_tok1, direction_vec_tok1)
                raise IndexError

            direction_vec_tok += direction_vec_tok1
            vertical_tok += vertical_tok1
            char_size_tok += csz_tok1

            # move pnt to in front of header-token and do nothing.
            r_pnt += 1
            c_pnt += 1

            # update header_tok
            for i in range(l_tok1 - 1):
                header_tok = np.insert(header_tok, obj=c_pnt + 1, values=0, axis=0)
            # upate label_sub
            label_type = ["f", "g", "root"]
            for i_label, label_sub1 in enumerate(label_sub):
                label_sub1, r_pnt_out, c_pnt_out = du.update_label_sub(
                    l_tok1, rel_idx, r_pnt, c_pnt, label_sub1, label_type[i_label]
                )
                label_sub[i_label] = label_sub1
            r_pnt = r_pnt_out
            c_pnt = c_pnt_out

        label_tok = label_sub

        return (
            text_tok,
            coord_tok,
            direction_vec,
            direction_vec_tok,
            vertical_tok,
            char_size_tok,
            label_tok,
            header_tok,
        )

    def get_center(self, coord):
        center = []
        for coord1 in coord:
            center1 = np.sum(coord1, axis=0) / 4
            center.append(center1)
        return center

    def get_angle(self, vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)

        angle = np.arccos(np.clip(np.dot(u1, u2), -1, 1))
        return angle

    def gen_rel_position(self, coord, direction_vec):
        coord = np.array(coord)
        center = self.get_center(coord)
        n_box = len(coord)
        rel_center = []
        rel_angle = np.zeros([n_box, n_box])

        for i_box in range(n_box):
            base_direction_vec1 = direction_vec[i_box]
            base_center1 = center[i_box]

            if self.omit_angle_cal:
                pass
            else:
                rel_angle[i_box, i_box:] = [
                    self.get_angle(base_direction_vec1, direction_vec1)
                    for direction_vec1 in direction_vec[i_box:]
                ]

            rel_center.append(center - base_center1)

        idx_lower = np.tril_indices(n_box, -1)
        rel_angle[idx_lower] = rel_angle.T[idx_lower]

        rel_center = np.stack(rel_center, axis=0)
        rel_dist = np.linalg.norm(rel_center, axis=-1)
        rel_angle = np.array(rel_angle)

        return rel_center, rel_dist, rel_angle

    def normalize(self, rel_center, rel_dist, rel_angle, img_sz, char_size_tok):
        rn_center = du.dist_normalization(
            self.dist_norm, self.n_dist_unit, rel_center, img_sz, char_size_tok
        )
        rn_dist = du.dist_normalization(
            self.dist_norm,
            self.n_dist_unit,
            rel_dist,
            img_sz,
            char_size_tok,
            all_positive=True,
        )
        rn_angle = du.angle_normalization(self.n_angle_unit, rel_angle)

        return rn_center, rn_dist, rn_angle
