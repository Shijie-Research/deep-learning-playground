"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._config import BaseConfig  # noqa
from .workspace import GLOBAL_CONFIG, create, register  # noqa
from .yaml_config import YAMLConfig  # noqa
from .yaml_utils import load_config, merge_config, merge_dict, parse_cli  # noqa
