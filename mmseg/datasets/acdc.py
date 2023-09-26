# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# 改编 多增一个day
# FDA后（如果后缀不变就不改）
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class ACDCDataset_day(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCDataset_day, self).__init__(
            img_suffix='_rgb_ref_anon.png',   #改编
            #img_night_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)

@DATASETS.register_module()
class ACDCDataset_night(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCDataset_night, self).__init__(
            img_suffix='_rgb_anon.png',   #改编  #改编
            #img_night_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)

