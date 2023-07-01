#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.utils.data.dataloader import DataLoader

from toolbox.wenet.models.model import Model
from toolbox.wenet.training.trainers.trainer import TrainerBase
from toolbox.wenet.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from toolbox.wenet.data.collate_functions.collate_fn import CollateFunction


logger = logging.getLogger(__name__)


class PyTorchLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model: nn.Module = None
        self.train_data_loader: DataLoader = None
        self.valid_data_loader: DataLoader = None

        self.optimizer: torch.optim.Optimizer = None

        # torch.optim.lr_scheduler._LRScheduler
        self.lr_scheduler = None

    def set_model(self, model: nn.Module):
        self.model = model

    def set_train_data_loader(self, train_data_loader: DataLoader):
        self.train_data_loader = train_data_loader

    def set_valid_data_loader(self, valid_data_loader: DataLoader):
        self.valid_data_loader = valid_data_loader

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def forward(self, *args, **kwargs) -> Any:
        outputs = self.model.forward(*args, **kwargs)
        return outputs

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.valid_data_loader

    def training_step(self, batch, batch_idx) -> Union[torch.Tensor, Dict[str, Any]]:
        outputs = self.forward(*batch)
        return outputs

    def training_step_end(self, step_output):
        # accuracy = self._accuracy.get_metric()
        for k, v in step_output.items():
            if k == 'loss':
                continue
            self.log(k, v, prog_bar=True)
        return step_output

    def training_epoch_end(self, outputs):
        return None

    def validation_step(self, batch, batch_idx) -> Union[torch.Tensor, Dict[str, Any]]:
        outputs = self.forward(*batch)
        return outputs

    def validation_step_end(self, step_output):
        # accuracy = self._accuracy.get_metric()
        for k, v in step_output.items():
            if k == 'loss':
                continue
            self.log('val_{}'.format(k), v, prog_bar=True)
        return step_output

    def validation_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            result = {
                'optimizer': self.optimizer,
            }
        else:
            result = {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': self.lr_scheduler
                },
            }
        return result


@TrainerBase.register('pytorch_lightning')
class PytorchLightningTrainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 train_dataset: Dataset,
                 train_collate_fn: CollateFunction,
                 valid_dataset: Dataset = None,
                 valid_collate_fn: CollateFunction = None,

                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",

                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 max_epochs: Optional[int] = None,
                 min_epochs: Optional[int] = None,
                 batch_size: int = 64,

                 accumulate_grad_batches: int = 1,

                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,

                 gradient_clip_val: Optional[float] = None,
                 gradient_clip_algorithm: str = 'norm',

                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,

                 log_every_n_steps: int = 10,
                 ):
        super().__init__(serialization_dir, cuda_device)
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.train_collate_fn = train_collate_fn
        self.valid_dataset = valid_dataset
        self.valid_collate_fn = valid_collate_fn

        self.patience = patience
        self.validation_metric = validation_metric
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

        self.num_serialized_models_to_keep = num_serialized_models_to_keep

        self.model_save_interval = model_save_interval

        self.cuda_device = cuda_device

        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.learning_rate_scheduler = learning_rate_scheduler

        self.log_every_n_steps = log_every_n_steps

        # pytorch lightning
        self.pytorch_lightning_model = PyTorchLightningModel()

        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=2,
        )
        valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=2,
        )

        # model
        self.pytorch_lightning_model.set_model(model)
        self.pytorch_lightning_model.set_train_data_loader(train_data_loader)
        self.pytorch_lightning_model.set_valid_data_loader(valid_data_loader)
        self.pytorch_lightning_model.set_optimizer(optimizer)
        self.pytorch_lightning_model.set_lr_scheduler(learning_rate_scheduler)

        # trainer
        callbacks = list()
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            monitor=self.validation_metric[1:],
            save_top_k=self.num_serialized_models_to_keep,
            mode='max' if self.validation_metric[0] else 'min',
        )
        callbacks.append(ckpt_callback)
        if self.patience is not None and self.patience > 0:
            early_stopping = pl.callbacks.EarlyStopping(
                monitor=self.validation_metric[1:],
                patience=self.patience,
                mode='max' if self.validation_metric[0] else 'min',
            )
            callbacks.append(early_stopping)

        self.pytorch_lightning_trainer = Trainer(
            callbacks=callbacks,
            default_root_dir=self._serialization_dir,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,

            # https://mp.weixin.qq.com/s?__biz=MzI1MjQ2OTQ3Ng==&mid=2247561650&idx=1&sn=ea6de6d2a6e4831c735d98d37cbfd026&chksm
            gpus=self.cuda_device,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            log_every_n_steps=self.log_every_n_steps,
            profiler='simple',

        )

    def train(self) -> Dict[str, Any]:
        return self.pytorch_lightning_trainer.fit(self.pytorch_lightning_model)
