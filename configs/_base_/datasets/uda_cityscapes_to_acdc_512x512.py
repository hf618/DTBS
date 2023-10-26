# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
# 数据集处理改编为两个targets
# FDA项目开启的数据源修改
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
acdc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(960, 540)),  # original 1920x1080
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 540),  # original 1920x1080
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train', #改
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        # type 对应的是 mmseg里的class ACDCDataset(CityscapesDataset)
        target_night=dict(
            type='ACDCDataset_night',
            data_root='data/acdc/',
            img_dir='rgb_anon/night/train',#autodl-tmp/DAFormer/data/acdc/rgb_anon/night/train/GOPR0351
            ann_dir='gt/night/train',#autodl-tmp/DAFormer/data/acdc/gt/night/train/GOPR0351
            pipeline=acdc_train_pipeline),
        target_day=dict(
            type='ACDCDataset_day',
            data_root='data/acdc/',
            img_dir='rgb_anon/night/train_ref',#autodl-tmp/DAFormer/data/acdc/rgb_anon/night/train_ref/GOPR0351
            ann_dir='gt/night/train',#autodl-tmp/DAFormer/data/acdc/gt/night/train/GOPR0351
            pipeline=acdc_train_pipeline)
    
    
    ),
    
    val=dict(
        type='ACDCDataset_night',
        data_root='data/acdc/',
        img_dir='rgb_anon/night/val',
        ann_dir='gt/night/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset_night',
        data_root='data/acdc/',
        img_dir='rgb_anon/night/val',
        ann_dir='gt/night/val',
        pipeline=test_pipeline))
'''
昨天晚上版本
     val=dict(
        type='ACDCDataset_day',
        data_root='data/acdc/',
        img_dir='rgb_anon/night/train_FDA2/GOPR0351',
        ann_dir='gt/night/train/GOPR0351',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset_day',
        data_root='data/acdc/',
        img_dir='rgb_anon/night/train_FDA2/GOPR0351',
        ann_dir='gt/night/train/GOPR0351',
        pipeline=test_pipeline))
'''   
'''

    val=dict(
        type='ACDCDataset_day',
        data_root='data/acdc/',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset',
        data_root='data/acdc/',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline))
'''    

