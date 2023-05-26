_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py']

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=20
        )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        dataset=dict(
            ann_file=['data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'],
            img_prefix=['data/VOCdevkit/VOC2012/']
        )
    ),
    val=dict(
        ann_file='data/VOCdevkit/VOC2012/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2012/'
    ),
    test=dict(
        ann_file='data/VOCdevkit/VOC2012/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2012/'
    )
)

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)

auto_scale_lr = dict(enable=False, base_batch_size=1)