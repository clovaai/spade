# SPADE
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import os
from copy import deepcopy
from pathlib import Path

import munch

import spade.utils.general_utils as gu


class ConfigManager(object):
    def __init__(self, config_dir, config_file_name):
        self.config_dir = config_dir
        self.config_file_name = config_file_name
        self.cfg = gu.load_yaml(Path(config_dir) / config_file_name)

        self.cfg = munch.munchify(self.cfg)
        self.cfg.path_data_folder = self._get_path_data_folder()
        if self.cfg.model_param.weights.trained:
            self.cfg.model_param.path_trained_model = self._get_path_trained_model(
                self.cfg.path_data_folder, self.cfg.model_param.weights.path
            )
            self.cfg.model_param.path_analysis_dir = (
                Path(os.path.dirname(self.cfg.model_param.path_trained_model))
                / "analysis"
                / config_file_name
            )

        self.cfg.config_file_name = config_file_name
        self.cfg.train_param.path_save_model_dir = (
            self.cfg.path_data_folder / "model" / "saved" / config_file_name
        )

    @staticmethod
    def _get_path_data_folder():
        path_data_repo = Path("./data")
        return path_data_repo

    @staticmethod
    def _get_path_trained_model(path_data_folder, weight_path):
        path_trained_model = Path(path_data_folder) / Path(weight_path)

        return path_trained_model
