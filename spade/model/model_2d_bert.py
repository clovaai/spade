# Based on modeling_bert.py from huggingface transformers.

import math

import pytorch_lightning as pl
import torch
import transformers
from torch import nn
from transformers.activations import ACT2FN


class TwoDimBertEncoder(pl.LightningModule):
    """
    The variable names from transformers.models.bert.modeling_bert.BertEncoder
    Other classes also follow the same convention to load the pretrained weights.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [TwoDimBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, rn_emb, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, rn_emb, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TwoDimBertLayer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.attention = TwoDimBertAttention(config)
        self.intermediate = TwoDimBertIntermediate(config)
        self.output = TwoDimBertOutput(config)

    def forward(self, hidden_states, rn_emb, attention_mask):
        attention_output = self.attention(hidden_states, rn_emb, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TwoDimBertAttention(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.self = TwoDimBertSelfAttention(config)
        self.output = TwoDimBertSelfOutput(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, input_tensor, rn_emb, attention_mask):
        if self.pre_layer_norm:
            input_tensor = self.LayerNorm(input_tensor)
        self_output = self.self(input_tensor, rn_emb, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class TwoDimBertSelfAttention(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.no_rel_attention = config.no_rel_attention
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sqrt_of_attention_hidden_size = math.sqrt(self.attention_head_size)
        if not self.no_rel_attention:
            # concat
            assert self.all_head_size == config.hidden_size
            assert self.all_head_size % 4 == 0
            quarter_of_hidden_size = int(self.all_head_size / 4)

            self.rn_dist = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)
            self.rn_angle = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)
            self.rn_center_x = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)
            self.rn_center_y = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)

            self.query_rel = nn.Linear(config.hidden_size, self.all_head_size)

            self.key_bias = nn.Parameter(
                torch.zeros(
                    config.num_attention_heads,
                    self.attention_head_size,
                )
            )
            self.rel_bias = nn.Parameter(
                torch.zeros(
                    config.num_attention_heads,
                    self.attention_head_size,
                )
            )

        # self.initializer_range = 0.02
        # self.key_bias = nn.Parameter(
        #     torch.randn(config.num_attention_heads, self.attention_head_size) * self.initializer_range)
        # self.rel_bias = nn.Parameter(
        #     torch.randn(config.num_attention_heads, self.attention_head_size) * self.initializer_range)

    def transpose_for_scores(self, x):
        """From this,"""
        # x = [B, seq_len, hidden_size]
        # [B, seq_len] + [num_attention_heads, attention_head_size]
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )

        # [B, seq_len, hidden_size] -> [B, seq_len, num_attention_heads, attention_head_size]
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)  # [B, Att_Head, seq_len, dim]

    def transpose_for_scores_rn(self, x):
        """From this,"""
        # x = [B, seq_len, hidden_size]
        # [B, seq_len] + [num_attention_heads, attention_head_size]
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )

        # [B, seq_len, hidden_size] -> [B, seq_len, num_attention_heads, attention_head_size]
        x = x.view(*new_x_shape)

        return x.permute(0, 3, 1, 2, 4)

    def forward(self, hidden_states, rn_emb, attention_mask):
        rn_center_x_emb, rn_center_y_emb, rn_dist_emb, rn_angle_emb = rn_emb
        # embedding

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if not self.no_rel_attention:
            rn_center_x_emb = self.rn_center_x(rn_center_x_emb)
            rn_center_y_emb = self.rn_center_y(rn_center_y_emb)
            rn_dist_emb = self.rn_dist(rn_dist_emb)
            rn_angle_emb = self.rn_angle(rn_angle_emb)

            rn_emb_all = torch.cat(
                [rn_center_x_emb, rn_center_y_emb, rn_dist_emb, rn_angle_emb], dim=-1
            )
            # mixed_rn_all_layer = self.rn_all(rn_emb_all)
            mixed_rn_all_layer = rn_emb_all
            rn_all_layer = self.transpose_for_scores_rn(mixed_rn_all_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.einsum("bhid,bhjd->bhij", query_layer, key_layer)

        # rn_dist
        if not self.no_rel_attention:
            mixed_query_rel_layer = self.query_rel(hidden_states)
            query_rel_layer = self.transpose_for_scores(mixed_query_rel_layer)

            attention_q_rn_all = torch.einsum(
                "bhid,bhijd->bhij", query_rel_layer, rn_all_layer
            )
            attention_scores += attention_q_rn_all

            key_bias = self.key_bias.unsqueeze(0).unsqueeze(2).expand_as(key_layer)
            rel_bias = self.rel_bias.unsqueeze(0).unsqueeze(2).expand_as(key_layer)

            attention_key_bias = torch.einsum("bhid,bhjd->bhij", key_bias, key_layer)
            attention_scores += attention_key_bias
            #
            attention_rel_bias = torch.einsum(
                "bhid,bhijd->bhij", rel_bias, rn_all_layer
            )
            attention_scores += attention_rel_bias

        attention_scores = attention_scores / self.sqrt_of_attention_hidden_size
        attention_scores = (
            attention_scores + attention_mask
        )  # sort of multiplication in soft-max step. It is ~ -10000

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # [B, num_attention_heads, seq_len, seq_len] * [B, num_attention_heads, seq_len, attention_head_size]
        # -> [B, num_attention_heads, seq_len, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # -> [B, seq_len, num_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # [B, seq_len] + [all_head_size=hidden_size] -> [B, seq_len, all_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class TwoDimBertSelfOutput(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor
        if not self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TwoDimBertIntermediate(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, hidden_states):
        if self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TwoDimBertOutput(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if not self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
