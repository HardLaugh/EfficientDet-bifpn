# model settings

model = dict(
    type='RetinaNet',
    #pretrained=None,
    pretrained='/home/46799/mmdetection/weights/efficientnet-b3-5fb5a3c3.pth',
    # pretrained='/home/46799/mmdetection/weights/efficientnet-b0-355c32eb.pth',
    # pretrained='/home/46799/mmdetection/weights/resnet101-5d3b4d8f.pth',
    backbone=dict(
        type='EfficientNet',
        arch='efficientnet-b0',
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[40, 112, 320], # efficientnet-b0
        # in_channels=[32, 48, 136, 384], # efficientnet-b3
        out_channels=64,
        num_outs=5,
        start_level=0,
        end_level=-1,
        add_extra_convs=True,
        stack=1,
        norm_cfg=dict(type='BN', momentum=0.003, eps=1e-4, requires_grad=False),
        activation='relu'),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=81,
        in_channels=64,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5, # 2.0 -> 1.5
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.10, loss_weight=50 / 4.)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'HeatCocoDataset'
data_root = '/home/46799/mmdetection/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(896, 896), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_garbage_train2019_1109.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_garbage_val2019_1109.json',
        img_prefix=data_root + 'validation/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_garbage_test2019_1109.json',
        img_prefix=data_root + 'validation/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=4e-5)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/coco_anchor_bifpn'
load_from = None
resume_from = None
auto_resume = True
workflow = [('train', 1)]
