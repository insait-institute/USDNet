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
import torch.nn.functional as F
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools
from models.articulate_graph_predict import ArticulationGraph

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

class InstSegArtGraph(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"
            
        self.config = config
        self.save_hyperparameters()
        # model
        self.mov_model = hydra.utils.instantiate(config.mov_model)
        self.inter_model = hydra.utils.instantiate(config.inter_model)
        self.arti_graph_model = ArticulationGraph(
            pcd_feats_size=self.mov_model.backbone.PLANES[7],
        )
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=True, matcher=None, use_gt_movable_mask = False, mov_target=None, inter_target = None, use_gt_movable_mask_as_pred=False
    ):
        with self.optional_freeze():
            # do inference for the movable and interactable segs
            mov_segs = self.mov_model(
                x,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
                matcher=matcher,
                target=mov_target,
                use_gt_movable_mask_as_pred = use_gt_movable_mask_as_pred
            )
            inter_segs = self.inter_model(
                x,
                point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
                matcher=matcher,
                use_gt_movable_mask = use_gt_movable_mask,
                target=inter_target,
                use_gt_movable_mask_as_pred = use_gt_movable_mask_as_pred
            )
            
        arti_graph_output = self.arti_graph_model(
            coords = x.coordinates,
            mov_pcd_feats = mov_segs['backbone_features'],
            mov_predictions = mov_segs, 
            mov_targets = mov_target,
            inter_pcd_feats = inter_segs['backbone_features'],
            inter_predictions = inter_segs,
            inter_targets = inter_target)
                
        return arti_graph_output

    def training_step(self, batch, batch_idx):
        mov_target, inter_target, data, file_names = batch

        if len(mov_target) == 0 or len(inter_target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            arti_graph_output = self.forward(
                data,
                raw_coordinates=raw_coordinates,
                mov_target = mov_target,
                inter_target=inter_target,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        try:
            loss = None
            accuracy_list = []
            recall_list = []
            precision_list = []
            gt_connectivity = arti_graph_output['gt_connectivity']
            pred_connectivity = arti_graph_output['pred_connectivity']
            num_gt_mov_nodes = arti_graph_output['num_gt_mov_nodes']
            num_gt_inter_nodes = arti_graph_output['num_gt_inter_nodes']
            num_matched_mov_nodes = arti_graph_output['num_matched_mov_nodes']
            num_matched_inter_nodes = arti_graph_output['num_matched_inter_nodes']
            
            batch_size = len(gt_connectivity)
            for bid in range(batch_size):
                gt_connectivity_bid = gt_connectivity[bid]
                pred_connectivity_bid = pred_connectivity[bid]
                if loss is not None:
                    loss += F.mse_loss(pred_connectivity_bid, gt_connectivity_bid)
                else:
                    loss = F.mse_loss(pred_connectivity_bid, gt_connectivity_bid)
                    
                # 
                tp = (pred_connectivity_bid > 0.5) & (gt_connectivity_bid > 0.5)
                fp = (pred_connectivity_bid > 0.5) & (gt_connectivity_bid < 0.5)
                fn = (pred_connectivity_bid < 0.5) & (gt_connectivity_bid > 0.5)
                tn = (pred_connectivity_bid < 0.5) & (gt_connectivity_bid < 0.5)
                accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum())
                recall = tp.sum() / (tp.sum() + fn.sum())
                precision = tp.sum() / (tp.sum() + fp.sum())
                accuracy_list.append(accuracy)
                recall_list.append(recall)
                precision_list.append(precision)
            accuracy = sum(accuracy_list) / len(accuracy_list)
            recall = sum(recall_list) / len(recall_list)
            precision = sum(precision_list) / len(precision_list)
            
        except ValueError as val_err:
            raise val_err

        logs = {
            'train_loss': loss.detach().cpu().item(),
            'train_accuracy': accuracy.detach().cpu().item(),
            'train_recall': recall.detach().cpu().item(),
            'train_precision': precision.detach().cpu().item(),
        }

        self.log_dict(logs)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)


    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def eval_step(self, batch, batch_idx):
        mov_target, inter_target, data, file_names = batch

        if len(mov_target) == 0 or len(inter_target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            arti_graph_output = self.forward(
                data,
                raw_coordinates=raw_coordinates,
                mov_target = mov_target,
                inter_target=inter_target,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        try:
            loss = None
            accuracy_list = []
            recall_list = []
            precision_list = []
            gt_connectivity = arti_graph_output['gt_connectivity']
            pred_connectivity = arti_graph_output['pred_connectivity']
            num_gt_mov_nodes = arti_graph_output['num_gt_mov_nodes']
            num_gt_inter_nodes = arti_graph_output['num_gt_inter_nodes']
            num_matched_mov_nodes = arti_graph_output['num_matched_mov_nodes']
            num_matched_inter_nodes = arti_graph_output['num_matched_inter_nodes']
            
            batch_size = len(gt_connectivity)
            for bid in range(batch_size):
                gt_connectivity_bid = gt_connectivity[bid]
                pred_connectivity_bid = pred_connectivity[bid]
                if loss is not None:
                    loss += F.mse_loss(pred_connectivity_bid, gt_connectivity_bid)
                else:
                    loss = F.mse_loss(pred_connectivity_bid, gt_connectivity_bid)
                    
                # 
                tp = (pred_connectivity_bid > 0.5) & (gt_connectivity_bid > 0.5)
                fp = (pred_connectivity_bid > 0.5) & (gt_connectivity_bid < 0.5)
                fn = (pred_connectivity_bid < 0.5) & (gt_connectivity_bid > 0.5)
                tn = (pred_connectivity_bid < 0.5) & (gt_connectivity_bid < 0.5)
                accuracy = (tp.sum() + tn.sum()) / (tp.sum() + tn.sum() + fp.sum() + fn.sum())
                recall = tp.sum() / (tp.sum() + fn.sum())
                precision = tp.sum() / (tp.sum() + fp.sum())
                accuracy_list.append(accuracy)
                recall_list.append(recall)
                precision_list.append(precision)
            accuracy = sum(accuracy_list) / len(accuracy_list)
            recall = sum(recall_list) / len(recall_list)
            precision = sum(precision_list) / len(precision_list)
            
        except ValueError as val_err:
            raise val_err

        logs = {
            'val_loss': loss.detach().cpu().item(),
            'val_accuracy': accuracy.detach().cpu().item(),
            'val_recall': recall.detach().cpu().item(),
            'val_precision': precision.detach().cpu().item(),
        }

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)


    def test_epoch_end(self, outputs):


        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():  # .items() in Python 3.
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}

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
        self.labels_info = self.train_dataset.label_info

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
