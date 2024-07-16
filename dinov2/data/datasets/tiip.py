# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset
import pathlib


logger = logging.getLogger("dinov2")
_Target = int


class Tiip(ExtendedVisionDataset):

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.data = sorted(pathlib.Path(root).rglob('*.jpg'))

    def get_image_data(self, index: int) -> bytes:
        image_full_path = str(self.data[index])
        # print(image_full_path)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> int:
        return 0 

    def __len__(self) -> int:
        return len(self.data)
