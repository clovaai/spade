# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

from pathlib import Path

import pytorch_lightning as pl
import torch
import transformers
from torch import nn

import spade.model.model_utils as mu
from spade.model.model_2d_bert import TwoDimBertEncoder


class SpadeEncoder(pl.LightningModule):
    def __init__(self, hparam, path_data_folder):
        super().__init__()
        # Augment config
        transformer_cfg = self.get_transformer_config(
            hparam.encoder_config_name,
            hparam.encoder_backbone_tweak_tag,
            path_data_folder,
        )
        transformer_cfg.pre_layer_norm = hparam.pre_layer_norm
        transformer_cfg.no_rel_attention = hparam.no_rel_attention
        transformer_cfg.trainable_rel_emb = hparam.trainable_rel_emb

        self.transformer_cfg = transformer_cfg

        self.embeddings = SpadeInputEmbeddings(
            transformer_cfg,
            hparam.n_dist_unit,
            hparam.n_char_unit,
            hparam.input_embedding_components,
        )

        if hparam.encoder_backbone_name in ["bert-base-multilingual-cased"]:
            self.encoder = TwoDimBertEncoder(transformer_cfg)
        else:
            raise NotImplementedError

        # rn embedding. rn stands for "relative, normalized"
        n_pos = hparam.n_dist_unit * 2 + 1
        self.n_dist_unit = hparam.n_dist_unit

        # check dimension compatibility
        assert transformer_cfg.hidden_size % 4 == 0
        quater_of_hidden_size = int(transformer_cfg.hidden_size / 4)
        if hparam.trainable_rel_emb:
            self.rn_center_x_emb = nn.Embedding(
                n_pos,
                quater_of_hidden_size,
                _weight=torch.zeros(n_pos, quater_of_hidden_size),
            )
            self.rn_center_y_emb = nn.Embedding(
                n_pos,
                quater_of_hidden_size,
                _weight=torch.zeros(n_pos, quater_of_hidden_size),
            )
            self.rn_angle_emb = nn.Embedding(
                hparam.n_angle_unit,
                quater_of_hidden_size,
                _weight=torch.zeros(hparam.n_angle_unit, quater_of_hidden_size),
            )
            self.rn_dist_emb = nn.Embedding(
                hparam.n_dist_unit,
                quater_of_hidden_size,
                _weight=torch.zeros(hparam.n_dist_unit, quater_of_hidden_size),
            )
        else:
            self.rn_center_x_emb = mu.SinCosPositionalEncoding(quater_of_hidden_size)
            self.rn_center_y_emb = mu.SinCosPositionalEncoding(quater_of_hidden_size)
            self.rn_angle_emb = mu.SinCosPositionalEncoding(quater_of_hidden_size)
            self.rn_dist_emb = mu.SinCosPositionalEncoding(quater_of_hidden_size)

    def get_transformer_config(
        self, encoder_config_name, encoder_backbone_tweak_tag, path_data_folder
    ):
        path_config = (
            Path(path_data_folder)
            / "model"
            / "backbones"
            / encoder_config_name
            / encoder_backbone_tweak_tag
            / "config.json"
        )
        transformer_cfg = transformers.BertConfig.from_json_file(path_config)
        return transformer_cfg

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
        token_type_ids=None,
        attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(text_tok_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        input_vectors = self.embeddings(
            text_tok_ids,
            rn_center_x_toks,  # [B, len, len]
            rn_center_y_toks,
            vertical_toks,  # [B, len]
            char_size_toks,  # [B, len]
            header_toks,  # [B, len]
            token_type_ids=token_type_ids,  # [B, len]
        )
        rn_emb = self.get_rn_emb(
            rn_center_x_toks,  # [B, len, len]
            rn_center_y_toks,
            rn_dist_toks,  # [B, len, len]
            rn_angle_toks,
        )
        all_encoder_layers = self.encoder(
            input_vectors, rn_emb, extended_attention_mask
        )
        # sequence_output = all_encoder_layers[-1]

        return all_encoder_layers

    def get_rn_emb(self, x, y, dist, angle):
        if self.transformer_cfg.trainable_rel_emb:
            return (
                self.rn_center_x_emb(x + self.n_dist_unit),
                self.rn_center_y_emb(y + self.n_dist_unit),
                self.rn_dist_emb(dist),
                self.rn_angle_emb(angle),
            )
        else:
            return (
                self.rn_center_x_emb(x),
                self.rn_center_y_emb(y),
                self.rn_dist_emb(dist),
                self.rn_angle_emb(angle),
            )


class SpadeInputEmbeddings(pl.LightningModule):
    """Based on BertEmbeddings of hugginface's"""

    def __init__(self, cfg, n_dist_unit, n_char_unit, input_embedding_components):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)

        # 2D embedding in input
        assert isinstance(input_embedding_components, list)
        self.input_embedding_components = input_embedding_components
        if "seqPos" in self.input_embedding_components:
            self.position_embeddings = nn.Embedding(
                cfg.max_position_embeddings, cfg.hidden_size
            )

            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer(
                "position_ids",
                torch.arange(cfg.max_position_embeddings).expand((1, -1)),
            )
            self.position_embedding_type = getattr(
                cfg, "position_embedding_type", "absolute"
            )

        self.n_dist_unit = n_dist_unit
        n_pos = n_dist_unit * 2 + 1

        if "absPos" in self.input_embedding_components:
            print("abs position added in the input")
            self.pos_x_embeddings = nn.Embedding(
                n_pos, cfg.hidden_size, _weight=torch.zeros(n_pos, cfg.hidden_size)
            )
            self.pos_y_embeddings = nn.Embedding(
                n_pos, cfg.hidden_size, _weight=torch.zeros(n_pos, cfg.hidden_size)
            )
        if "charSize" in self.input_embedding_components:
            self.char_size_embeddings = nn.Embedding(
                n_char_unit,
                cfg.hidden_size,
                _weight=torch.zeros(n_char_unit, cfg.hidden_size),
            )
        if "vertical" in self.input_embedding_components:
            self.vertical_embeddings = nn.Embedding(
                2, cfg.hidden_size, _weight=torch.zeros(2, cfg.hidden_size)
            )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(
        self,
        text_tok_ids,
        rn_center_x_ids,
        rn_center_y_ids,
        vertical_ids,
        char_size_ids,
        header_ids,
        token_type_ids=None,
    ):

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(text_tok_ids)

        words_embeddings = self.word_embeddings(text_tok_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        if "seqPos" in self.input_embedding_components:
            seq_length = text_tok_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=text_tok_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(text_tok_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if "absPos" in self.input_embedding_components:
            pos_x_embeddings = self.pos_x_embeddings(
                rn_center_x_ids[:, 0] + self.n_dist_unit
            )  # use first token as an origin
            pos_y_embeddings = self.pos_y_embeddings(
                rn_center_y_ids[:, 0] + self.n_dist_unit
            )

            embeddings += pos_x_embeddings
            embeddings += pos_y_embeddings

        if "charSize" in self.input_embedding_components:
            char_size_embeddings = self.char_size_embeddings(vertical_ids)
            embeddings += char_size_embeddings
            # print('cc')
        if "vertical" in self.input_embedding_components:
            vertical_embeddings = self.vertical_embeddings(vertical_ids)
            embeddings += vertical_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
