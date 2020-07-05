# fp16 setting
# fp16 = dict(loss_scale=512.)
# model settings
norm_cfg=dict(type='SyncBN', momentum=0.01,
                eps=1e-3, requires_grad=True)
model = dict(
    type='RetinaNet',
    pretrained='efficientnet-b2-8bb594d6.pth',
    backbone=dict(
        type='EfficientNet',
        arch='efficientnet-b2',
        out_levels=[3, 4, 5],
        norm_cfg=norm_cfg,
        norm_eval=False,
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[48, 120, 352],
        out_channels=112,
        num_outs=5,
        strides=[8, 16, 32],
        start_level=0,
        end_level=-1,
        stack=5,
        norm_cfg=norm_cfg,
        act_cfg=None),
    bbox_head=dict(
        type='RetinaSepConvHead',
        num_classes=90,
        in_channels=112,
        stacked_convs=3,
        feat_channels=112,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,  # 2.0 -> 1.5
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='HuberLoss', beta=0.1, loss_weight=50.)
        # loss_bbox=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.)
))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=5000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# dataset settings
dataset_type = 'CocoDataset'
data_root = './coco/'
size = 768
img_scale = (size, size)
ratio_range = (0.1, 2.0)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=img_scale,
         keep_ratio=True, ratio_range=ratio_range),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomCrop', crop_size=(size, size)),
    dict(type='Pad', size=(size, size)),
    # dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        # img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomFlip'),
            # dict(type='Pad', size_divisor=128),
            dict(type='Pad', size=(size, size)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=4e-5)
# optimizer = dict(type='Adam', lr=1e-3)

optimizer_config = dict(grad_clip=dict(max_norm=10.0, norm_type=2))

# yapf:enable

interval = 1
out_dir = './work_dirs/coco_b2-baseline'
evaluation = dict(interval=interval)
# runtime settings
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = out_dir
load_from = None
resume_from = None
resume_from = './work_dirs/coco_b2-baseline/latest.pth'
auto_resume = True
workflow = [('train', 1)]


# learning policy
lr_config = dict(
    policy='CosineAnealing',
    min_lr = 0.0001,
    min_lr_ratio=None,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=0.0001)

checkpoint_config = dict(interval=interval)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# EMA config
ema_config = dict(
    use_ema=True,
    decay=0.9998,
    out_dir=out_dir,
    interval=interval,
    device='',
)




