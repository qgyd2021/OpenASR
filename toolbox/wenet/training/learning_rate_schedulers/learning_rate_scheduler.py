#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, List
import torch

from toolbox.wenet.common.registrable import Registrable
from toolbox.wenet.training.scheduler import Scheduler


class LearningRateScheduler(Scheduler, Registrable):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    def get_values(self) -> None:
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):

    def __init__(self,
                 lr_scheduler,   # torch.optim.lr_scheduler._LRScheduler
                 ) -> None:
        super().__init__(optimizer=self.lr_scheduler.optimizer)
        self.lr_scheduler = lr_scheduler

    def get_values(self):
        return self.lr_scheduler.get_lr()

    def step(self, metric: float = None, epoch: int = None) -> None:
        self.lr_scheduler.step(epoch)

    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):

    def step(self, metric: float = None, epoch: int = None) -> None:
        if metric is None:
            raise AssertionError('This learning rate scheduler requires a validation '
                                 'metric to compute the schedule and therefore must be '
                                 'used with a validation dataset.')
        self.lr_scheduler.step(metric, epoch)


Registrable._registry[LearningRateScheduler] = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multi_step": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}
