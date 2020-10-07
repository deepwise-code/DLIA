# -*- coding: utf-8 -*-
"""
   Author :       lirenqiang
   date：          2019/10/16
   Reference:
   PolynomialLR and WarmUpLR:
   https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/schedulers/schedulers.py
   StepLR:
   https://pytorch.org/docs/0.4.1/_modules/torch/optim/lr_scheduler.html
   https://pytorch.org/docs/1.0.0/optim.html

"""
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR
from bisect import bisect_right
from torch._six import inf
from functools import partial


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # keep same learn rate during how many iteration(decay_iter)
        if self.last_epoch % self.decay_iter != 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(
            self, optimizer, scheduler, mode="linear", warmup_iters=100, gamma=0.2, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.scheduler.last_epoch = self.last_epoch
        cold_lrs = self.scheduler.get_lr()
        if self.last_epoch < self.warmup_iters:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs


class StepLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        # >>> # Assuming optimizer uses lr = 0.05 for all groups
        # >>> # lr = 0.05     if epoch < 30
        # >>> # lr = 0.005    if 30 <= epoch < 60
        # >>> # lr = 0.0005   if 60 <= epoch < 90
        # >>> # ...
        # >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        # >>> for epoch in range(100):
        # >>>     scheduler.step()
        # >>>     train(...)
        # >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class MultiStepLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        # >>> # Assuming optimizer uses lr = 0.05 for all groups
        # >>> # lr = 0.05     if epoch < 30
        # >>> # lr = 0.005    if 30 <= epoch < 80
        # >>> # lr = 0.0005   if epoch >= 80
        # >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        # >>> for epoch in range(100):
        # >>>     scheduler.step()
        # >>>     train(...)
        # >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        # >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        # >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        # >>> for epoch in range(10):
        # >>>     train(...)
        # >>>     val_loss = validate(...)
        # >>>     # Note that step should be called after validate()
        # >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # if not isinstance(optimizer, Optimizer):
        #     raise TypeError('{} is not an Optimizer'.format(
        #         type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
