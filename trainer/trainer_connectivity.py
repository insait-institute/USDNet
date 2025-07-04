import gc
from contextlib import nullcontext
from pathlib import Path
import statistics
import shutil
import os
import math
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
import matplotlib
from benchmark.evaluate_semantic_instance import evaluate
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils.votenet_utils.eval_det import eval_det
from datasets.scannet200.scannet200_splits import (
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    VALID_CLASS_IDS_200_VALIDATION,
)

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools
from models.criterion_connectivity import loss_connectivity

@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")
        
class EpochSettingCallback(pl.Callback):
    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        current_epoch = trainer.current_epoch
        if hasattr(pl_module.train_dataloader().dataset, 'set_epoch'):
            pl_module.train_dataloader().dataset.set_epoch(current_epoch)

class ConnectivityPrediction(pl.LightningModule):
    def __init__(self, config):
        super().__init__()


        self.config = config
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad

        self.loss = loss_connectivity
        

    def forward(
        self, data
    ):
        with self.optional_freeze():
            x = self.model(
                data,
            )
        return x

    def training_step(self, batch, batch_idx):
        
        data = batch
        target = data['edge_matrix_list']
        output = None

        output = self.forward(
                data,
            )

        loss_dict = self.loss(output, target)


        logs = {
            "train_loss": loss_dict["loss"].detach().cpu().item(),
            "train_accuracy_edge": loss_dict["accuracy_edge"].detach().cpu().item(),
            "train_accuracy_object": loss_dict["accuracy_object"].detach().cpu().item(),
        }

        self.log_dict(logs)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        train_loss = sum([out['loss'] for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def eval_step(self, batch, batch_idx):
        data = batch
        target = data['edge_matrix_list']
        output = None
        output = self.forward(
                data,
            )

        loss_dict = self.loss(output, target)

        logs = {
            "val_loss": loss_dict["loss"].detach().cpu().item(),
            "val_accuracy_edge": loss_dict["accuracy_edge"].detach().cpu().item(),
            "val_accuracy_object": loss_dict["accuracy_object"].detach().cpu().item(),
        }

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):

        val_loss_mean = sum([out["val_loss"] for out in outputs]) / len(outputs)
        val_accuracy_edge_mean = sum([out["val_accuracy_edge"] for out in outputs]) / len(outputs)
        val_accuracy_object_mean = sum([out["val_accuracy_object"] for out in outputs]) / len(outputs)
        
        dd = {"val_loss_mean": val_loss_mean,
                "val_accuracy_edge_mean": val_accuracy_edge_mean,
                "val_accuracy_object_mean": val_accuracy_object_mean,
            }
        
        self.log_dict(dd, on_epoch=True)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(
            self.config.data.test_dataset
        )
        
        print("len of train dataset", len(self.train_dataset))
        print("len of val dataset", len(self.validation_dataset))
        print("len of test dataset", len(self.test_dataset))

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
