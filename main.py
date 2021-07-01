# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0
import argparse

from spade import Agent, ConfigManager


def main():
    args = get_args()
    cfg = ConfigManager(args.config_dir, args.config_file_name).cfg
    agent = Agent(cfg)
    if args.mode == "preprocess":
        agent.do_preprocess()
    elif args.mode == "train":
        agent.do_training()
    elif args.mode == "test":
        agent.do_testing()
    elif args.mode == "predict":
        agent.do_prediction(args.path_predict_input_json)
    else:
        raise ValueError


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_name")
    parser.add_argument(
        "-m", "--mode", help="preprocess|train|eval|make_resource|release|serve"
    )  # list
    parser.add_argument("-d", "--config_dir", default="./configs")
    parser.add_argument("-p", "--path_predict_input_json", default="")
    args = parser.parse_args()
    if args.path_predict_input_json:
        assert args.mode == "predict"
    return args


if __name__ == "__main__":
    main()
