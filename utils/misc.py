import os
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import logging.handlers
import torch.distributed as dist
import torch.optim as optim

from collections import OrderedDict




class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

    def save(self, model, args, cd, f1, fname='ckpt.pt', optimizer=None, epoch=None, dist=False):

        path = os.path.join(self.save_dir, fname)

        if dist:
            torch.save({
                'args': args,
                'state_dict': model.module.state_dict(),
                'cd': cd,
                'f1': f1,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, path)
        else:
            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
                'cd': cd,
                'f1': f1,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, path)


        return True



def setup(rank, world_size):
    torch.backends.cudnn.benchmark = True
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_optimizer(base_model, args):
    
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    param_groups = add_weight_decay(base_model, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def resume_optimizer(optimizer, ckpt_path):
    
    if not os.path.exists(ckpt_path):
        return 0, 0, 0
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])



def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)




def build_lambda_sche(opti, last_epoch=-1):
    
    # lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
    warming_up_t = 0
    lr_lbmd = lambda e: max(0.9 ** ((e - warming_up_t) / 21), 0.02) if e >= warming_up_t else max(e / warming_up_t, 0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    
    return scheduler


def build_lambda_bnsche(model, last_epoch=-1):
    bnm_lmbd = lambda e: max(0.9 * 0.5 ** (e / 21), 0.01)
    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    
    return bnm_scheduler


def build_scheduler(base_model, optimizer, last_epoch=-1, args=None):
    if args.sched == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, last_epoch=last_epoch)
    else:
        raise NotImplementedError("Not supported")
   
    bnscheduler = build_lambda_bnsche(base_model)
    scheduler = [scheduler, bnscheduler]
    
    return scheduler


def resume_model(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise ValueError("No ckpt found to resume")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    best_cd = ckpt['cd']
    best_f1 = ckpt['f1']
    last_epoch = ckpt['epoch']
    return best_cd, last_epoch



def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


