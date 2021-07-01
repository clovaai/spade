# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

# Exectue the file inside the "scripts" folder
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import transformers

import spade.utils.general_utils as gu


def download_cache(model_card):
    if model_card in ["xlm-roberta-base", "xlm-roberta-large"]:
        path_save_dir = Path(f"../data/model/backbones/{model_card}_lm")
        get_tokenizer = transformers.XLMRobertaTokenizer
        get_config = transformers.XLMRobertaConfig
        get_model = transformers.XLMRobertaForCausalLM
    elif model_card in ["facebook/mbart-large-cc25"]:
        path_save_dir = Path(f"../data/model/backbones/{model_card}")
        get_tokenizer = transformers.MBartTokenizer
        get_config = transformers.MBartConfig
        get_model = transformers.MBartForConditionalGeneration
    elif model_card in ["bert-base-multilingual-cased"]:
        path_save_dir = Path(f"../data/model/backbones/{model_card}")
        get_tokenizer = transformers.BertTokenizer
        get_config = transformers.BertConfig
        get_model = transformers.BertModel

    else:
        raise NotImplementedError
    path_save_cache_dir = path_save_dir / "cache"
    tokenizer = get_tokenizer.from_pretrained(model_card, cache_dir=path_save_cache_dir)
    model_config = get_config.from_pretrained(model_card, cache_dir=path_save_cache_dir)
    model = get_model.from_pretrained(
        model_card, config=model_config, cache_dir=path_save_cache_dir
    )

    return tokenizer, model_config, model, path_save_dir


def get_new_sepcial_toks(task):
    if task in ["spade"]:
        new_special_toks = {"additional_special_tokens": []}

    elif task in ["wyvern"]:
        new_special_toks = {
            "additional_special_tokens": [
                "<S>",
                "<T>",
                "[SPC]",
                "[reduce]",
                "<mobile>",
                "<full_name>",
                "<full_name_sort>",
                "<company>",
                "<company_sort>",
                "<department>",
                "<position>",
                "<address>",
                "<tel>",
                "<fax>",
                "<email>",
                "<phone>",
                "<homepage>",
                "<furigana>",
                "<paymentInfo>",
                "<cardInfo>",
                "<number>",
                "<confirmNum>",
                "<dtm>",
                "<storeInfo>",
                "<bizNum>",
                "<name>",
                "<subResults>",
                "<items>",
                "<priceInfo>",
                "<price>",
                "<unitPrice>",
                "<totalPrice>",
                "<subName>",
                "<list>",
                "<count>",
                "<complexMallName>",
                "<store mall>",
                "<store sub name>",
                "<store name>",
                "<store business number>",
                "<store address>",
                "<store tel>",
                "<store etc name>",
                "<store movie name>",
                "<store branch>",
                "<store branch number>",
                "<store etc>",
                "<payment date>",
                "<payment method>",
                "<payment time>",
                "<payment card company>",
                "<payment card number>",
                "<payment confirmed number>",
                "<payment no cash return>",
                "<payment change>",
                "<payment price>",
                "<payment person>",
                "<payment account number>",
                "<payment type>",
                "<payment bank code>",
                "<payment bank number>",
                "<payment branch code>",
                "<payment branch name>",
                "<payment due>",
                "<payment swift>",
                "<item tax info>",
                "<item name>",
                "<item unit price>",
                "<item count>",
                "<item currency>",
                "<item price>",
                "<item number>",
                "<item cut>",
                "<item etc>",
                "<item date>",
                "<item unit>",
                "<item tax>",
                "<item external price>",
                "<item external unit price>",
                "<item internal price>",
                "<item sub code>",
                "<item sub count>",
                "<item sub cut>",
                "<item sub etc>",
                "<item sub name>",
                "<item sub number>",
                "<item sub price>",
                "<item sub unit price>",
                "<sub total cut>",
                "<sub total other price>",
                "<sub total sub price>",
                "<sub total price>",
                "<sub total service price>",
                "<sub total tax>",
                "<sub total income tax>",
                "<total etc>",
                "<total count>",
                "<total price>",
                "<total tax>",
                "<transaction date>",
                "<transaction time>",
                "<transaction recipient>",
                "<transaction reason>",
                "<url>",
                "<Japanese name>",
                "<from address>",
                "<from company>",
                "<from department>",
                "<from email>",
                "<from fax>",
                "<from owner>",
                "<from person>",
                "<from postal>",
                "<from tel>",
                "<from url>",
                "<to address>",
                "<to company>",
                "<to department>",
                "<to email>",
                "<to fax>",
                "<to owner>",
                "<to person>",
                "<to postal>",
                "<to tel>",
                "<to url>",
                "<contact address>",
                "<contact department>",
                "<contact email>",
                "<contact fax>",
                "<contact person>",
                "<contact postal>",
                "<contact tel>",
                "<contact url>",
                "<general etc date>",
                "<general etc number>",
                "<general document number>",
                "<general closing date>",
                "<general issue date>",
                "<general subject>",
                "<general sub amount>",
                "<general tax info>",
                "<general total amount>",
                "<general total tax>",
                "<other note>",
            ]
        }
    else:
        raise NotImplementedError

    assert len(new_special_toks["additional_special_tokens"]) == len(
        set(new_special_toks["additional_special_tokens"])
    )
    return new_special_toks


def extend_vocab(tokenizer, model, new_special_tokens):
    _num_added_toks = tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(tokenizer))


def save_pretrained(path_save_dir, tokenizer, model_config, model, sub_folder_name):
    tokenizer.save_pretrained(path_save_dir / sub_folder_name)
    model_config.save_pretrained(path_save_dir / sub_folder_name)
    model.save_pretrained(path_save_dir / sub_folder_name)
    gu.cnt_model_weights(model)


def main(args):
    for model_card in args.model_cards:
        print(model_card)
        tokenizer, model_config, model, path_save_dir = download_cache(model_card)
        save_pretrained(path_save_dir, tokenizer, model_config, model, "org")

        # new vocab extended version
        # new_special_toks = get_new_sepcial_toks(args.task)
        # extend_vocab(tokenizer, model, new_special_toks)
        # save_pretrained(path_save_dir, tokenizer, model_config, model, "extended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.task = "spade"  # or "wyvern"

    model_card1 = "xlm-roberta-base"
    model_card2 = "xlm-roberta-large"
    model_card3 = "facebook/mbart-large-cc25"
    model_card4 = "bert-base-multilingual-cased"

    args.model_cards = [model_card4]
    main(args)
