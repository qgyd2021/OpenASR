#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Dict, Any

import torch


class Scheduler(object):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 last_epoch: int = -1
                 ) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = 'initial_{}'.format(param_group_field)
        if last_epoch == -1:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError('{} missing from param_groups[{}]'.format(param_group_field, i))
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError('{} missing from param_groups[{}]'.format(self._initial_param_group_field, i))
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.step(epoch=last_epoch)
        self.last_epoch = last_epoch

        self.metric = None

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_values(self):
        raise NotImplementedError

    def step(self, metric: float = None, epoch: int = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self.metric = metric
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = value

    def step_batch(self, batch_num_total: int = None) -> None:
        return None
