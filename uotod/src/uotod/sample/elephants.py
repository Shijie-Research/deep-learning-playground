import os

import numpy
import torch
from PIL import Image


_filedir = os.path.dirname(os.path.realpath(__file__))
_filename = "elephants.jpg"
_filepath = os.path.join(_filedir, _filename)

with Image.open(_filepath) as pil_img:
    img_elephants = numpy.asarray(pil_img)

boxes_pred_elephants = torch.tensor(
    [
        [130.0, 160.0, 320.0, 320.0],
        [350.0, 280.0, 550.0, 400.0],
        [105.0, 130.0, 265.0, 260.0],
        [300.0, 105.0, 570.0, 295.0],
        [20.0, 50.0, 120.0, 120.0],
    ]
)

cls_pred_elephants = torch.tensor(
    [
        [0.2, 0.7, 0.1],
        [0.1, 0.05, 0.85],
        [0.1, 0.05, 0.85],
        [0.2, 0.7, 0.1],
        [0.1, 0.05, 0.85],
    ],
    dtype=torch.float,
)

boxes_target_elephants = torch.tensor([[292.62, 134.59, 518.84, 285.05], [114.35, 148.97, 341.14, 297.63]])

cls_target_elephants = torch.tensor([0, 1], dtype=torch.long)

mask_target_elephants = torch.tensor([True, True], dtype=torch.bool)
