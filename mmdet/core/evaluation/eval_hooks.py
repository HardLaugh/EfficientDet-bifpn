import os.path as osp
import mmcv
import math
from copy import deepcopy

from mmcv.runner import Hook
from mmcv.runner.dist_utils import master_only, get_dist_info
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mmcv.runner.checkpoint import save_checkpoint, load_checkpoint

class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

        # EMA test
        # if runner.rank == 0:
        # results = multi_gpu_test(
        #     runner.ema.ema,
        #     self.dataloader,
        #     tmpdir=osp.join(runner.work_dir, '.eval_hook'),
        #     gpu_collect=self.gpu_collect)
        # if runner.rank == 0:
        #     print('\n')
        #     self.evaluate(runner, results)


class ModelEMA(Hook):
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, 
                runner,
                filename=None,
                decay=0.9998, 
                out_dir=None, 
                interval=-1, 
                save_optimizer=True, 
                max_keep_ckpts=-1, 
                meta=None,
                device='', 
                **kwargs):
        # setting checkpoints
        self.interval = interval
        self.out_dir = out_dir
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs
        self.create_symlink = True
        self.filename = filename
        self.meta = meta
        
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(runner.model)
        if self.filename:
            runner.logger.info('load EMA checkpoint for EMA model from %s', filename)
            load_checkpoint(self.ema, self.filename, map_location='cpu', strict=False)
        self.ema.eval()
        # self.updates = 0  # number of EMA updates
        self.updates = runner.iter  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

        runner.ema = self

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))

    # @master_only
    def after_train_iter(self, runner):
        # rank, _ = get_dist_info()
        # for test i dont use master_only
        self.update(runner.model)

    @master_only
    def after_train_epoch(self, runner):
        self.update_attr(runner.model)

        # save ema model
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir

        meta = runner.meta
        
        if meta is None:
            meta = dict(epoch=runner.epoch + 1, iter=runner.iter)
        else:
            meta.update(epoch=runner.epoch + 1, iter=runner.iter)

        filename = 'epoch_ema_{}.pth'.format(runner.epoch + 1)
        filepath = osp.join(self.out_dir, filename)
        optimizer = runner.optimizer if self.save_optimizer else None
        save_checkpoint(
            self.ema, 
            filepath,
            optimizer=optimizer, 
            meta=meta)
        if self.create_symlink:
            mmcv.symlink(filename, osp.join(self.out_dir, 'latest_ema.pth'))

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'epoch_ema_{}.pth')
            current_epoch = runner.epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(epoch))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break
