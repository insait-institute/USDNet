# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn


def loss_connectivity(
    inputs: torch.Tensor,
    targets: torch.Tensor
):
    """
    Args:
        inputs: a batchlist of a list of object edge connectivity, \in [-1, 1]
        targets: a batchlist of a list of object edge connectivity, \in (-1, 0, 1)
    Returns:
        Loss tensor
    """
    batch_size = len(inputs)
    assert batch_size == len(targets), f"batch_size: {batch_size}, len(targets): {len(targets)}"
    
    loss = None
    accuracy_edge = []
    accuracy_object = []
    num_objs_batch = 0
    for bid in range(batch_size):
        input = inputs[bid]
        target = targets[bid]
        num_objs = len(target)
        assert len(input) == num_objs, f"len(input): {len(input)}, num_objs: {num_objs}"
        
        num_objs_batch += num_objs
        
        for obj_idx in range(num_objs):
            input_obj = input[obj_idx].float()
            target_obj = target[obj_idx].float()
            assert input_obj.shape == target_obj.shape, f"input_obj.shape: {input_obj.shape}, target_obj.shape: {target_obj.shape}"
            if loss is not None:
                loss += F.mse_loss(input_obj, target_obj)  
            else:
                loss = F.mse_loss(input_obj, target_obj)
            
            # calculate success rate for each object
            predict_edge = torch.zeros_like(input_obj, dtype=torch.long)
            edge_confidence_matrix = input_obj - input_obj.T
            predict_edge[edge_confidence_matrix > 1] = 1
            predict_edge[edge_confidence_matrix < -1] = -1  
            accuracy_edge.append(  ((predict_edge == target_obj).sum() -predict_edge.shape[0])  / (target_obj.numel() -predict_edge.shape[0])  )
            accuracy_object.append((predict_edge == target_obj).all().float())
    out_dict = {
        "loss": loss / num_objs_batch,
        "accuracy_edge": sum(accuracy_edge) / len(accuracy_edge),
        "accuracy_object": sum(accuracy_object) / len(accuracy_object)
    }
    return out_dict

