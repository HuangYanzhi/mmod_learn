checkpoint_config = dict(interval=1)    # 1个epoch保存一次checkpoint
# yapf:disable
log_config = dict(  # register logger hook 的配置文件。
    interval=50,    # 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl') # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'
load_from = None    # 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
