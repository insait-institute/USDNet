# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def dice_loss_inspect(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    print("numerator: ", numerator)
    print("denominator: ", denominator)
    print("inputs: ", inputs)
    print("targets: ", targets)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule
dice_loss_inspect_jit = torch.jit.script(dice_loss_inspect)  # type: torch.jit.ScriptModule

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
        regular_arti_loss = True,
        use_mov_mask_for_interaction_loss = False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.regular_arti_loss = regular_arti_loss
        print("self.regular_arti_loss: ", self.regular_arti_loss)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        self.use_mov_mask_for_interaction_loss = use_mov_mask_for_interaction_loss
        
        self.debug_counter = 0

    def loss_labels(self, outputs, targets, indices, 
                    num_masks, mask_type, debug=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            ignore_index=253,
        )
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_articulations(self, outputs, targets, indices, 
                           num_masks, mask_type, debug=False):
        loss_origins = []
        loss_axises = []
        
        for batch_id, (map_ids, target_ids) in enumerate(indices):
            map_origins = outputs["pred_origins"][batch_id][map_ids, :]
            target_origins = targets[batch_id]['articulations']['origin'][target_ids]
            map_axises = outputs["pred_axises"][batch_id][map_ids, :]
            target_axises = targets[batch_id]['articulations']['axis'][target_ids]
            if map_origins.shape[0] == 0 or target_origins.shape[0] == 0:
                loss_origins.append(torch.tensor(0.0, device=map_origins.device))
                loss_axises.append(torch.tensor(0.0, device=map_origins.device))
                continue
            if self.regular_arti_loss:
                # print(f"Shape of outputs['pred_origins']: {outputs['pred_origins'].shape}")
                # print(f"Shape of targets[batch_id]['articulations']['axis'].shape: {targets[batch_id]['articulations']['axis'].shape}")
                # print(f"Batch ID: {batch_id}, Map ID: {map_id}, target ID: {target_id}")
                loss_origins.append(F.l1_loss(map_origins, 
                        target_origins, reduction='sum') / map_origins.shape[0])
                loss_axises.append(F.l1_loss(map_axises,
                        target_axises, reduction='sum') / map_axises.shape[0])
            else:
                # calculate axis loss
                map_axises_normalized = F.normalize(map_axises, dim=1)
                target_axises_normalized = F.normalize(target_axises, dim=1)
                loss_axises_vec = 1 - torch.sum(map_axises_normalized * target_axises_normalized, dim=1)
                loss_axises.append(torch.sum(loss_axises_vec) / map_axises_normalized.shape[0])
                
                # calculate origin loss                    
                loss_translations_item = []
                for idx, target_id in enumerate(target_ids):
                    map_id = map_ids[idx]
                    pred_origin = outputs["pred_origins"][batch_id][map_id, :].cuda()
                    pred_axis = outputs["pred_axises"][batch_id][map_id, :].cuda()
                    target_origin = targets[batch_id]['articulations']['origin'][target_id].cuda()
                    target_axis = targets[batch_id]['articulations']['axis'][target_id].cuda()
                    articulation_label = targets[batch_id]['labels'][target_id].cuda()
                    assert articulation_label == 1 or articulation_label == 2 ,"Articulation label is not valid: {}".format(articulation_label)
                    if articulation_label == 1:
                        # rotation
                        ## calculat the distance between pred_origin to the line defined by target_origin and target_axis
                        dist_vec_gt_axis = torch.cross(pred_origin - target_origin, target_axis)
                        dist_vec_pred_axis = torch.cross(target_origin - pred_origin, pred_axis)
                        # print("target_axis shape: ", target_axis.shape)
                        dist_gt_axis = torch.norm(dist_vec_gt_axis) / torch.norm(target_axis)
                        dist_pred_axis = torch.norm(dist_vec_pred_axis) / (torch.norm(pred_axis) + torch.tensor(1e-6, device=map_origins.device))
                        loss_translations_item.append(dist_gt_axis + dist_pred_axis)
                    else:
                        # translation, don't care about the origin
                        loss_translations_item.append(torch.tensor(0.0, device=map_origins.device))
                     
                loss_origins.append(torch.sum(torch.stack(loss_translations_item)) / len(loss_translations_item))
            
            # if debug:
            #     print(f"    Pred origin: {map_origins}; Target origin: {target_origins}")
        return {
            "loss_origin": torch.sum(torch.stack(loss_origins)),
            "loss_axis": torch.sum(torch.stack(loss_axises)),
        }
        
    def loss_interaction_masks(self, outputs, targets, indices,
                               num_masks, mask_type, debug=False):
        loss_masks = []
        loss_dices = []
        num_tp_all, num_fp_all, num_tp_gt, num_fp_gt, num_tp_pred, num_fp_pred = 0, 0, 0, 0, 0, 0

        for batch_id, (map_ids, target_ids) in enumerate(indices):
            num_masks = len(map_ids)
            pred_inter_masks = outputs["pred_interaction_dict"]['interaction_mask'][batch_id][:, map_ids].T # [num_masks, num_points]
            gt_inter_masks = targets[batch_id]['interaction_masks'][target_ids, :].float() # [ num_masks, num_points]
            
            if num_masks == 0:
                loss_masks.append(torch.tensor(0.0, device=pred_inter_masks.device))
                loss_dices.append(torch.tensor(0.0, device=pred_inter_masks.device))
            else:
                loss_ce_inter_mask = sigmoid_ce_loss_jit(pred_inter_masks, gt_inter_masks, num_masks)
                loss_dice_inter_mask = dice_loss_jit(pred_inter_masks, gt_inter_masks, num_masks)
                loss_masks.append(loss_ce_inter_mask )
                loss_dices.append(loss_dice_inter_mask)
                
        #         # for debug use, print precision
        #         if self.debug_counter % 20 == 0 and debug:
        #             gt_masks = targets[batch_id]['masks'][target_ids, :] # [ num_masks, num_points]
        #             pred_mov_masks = outputs["pred_masks"][batch_id][:, map_ids].T # [ num_masks, num_points]
                    
        #             for idx in range(num_masks):
        #                 gt_mask = gt_masks[idx]
        #                 pred_mov_mask = pred_mov_masks[idx].sigmoid() > 0.5
        #                 gt_inter_mask = gt_inter_masks[idx]> 0.5
        #                 pred_inter_mask_score = pred_inter_masks[idx]
        #                 pred_inter_mask = pred_inter_mask_score.sigmoid() > 0.5
                        
        #                 insec = (pred_inter_mask) & (gt_inter_mask)
        #                 union = (pred_inter_mask) | (gt_inter_mask)
        #                 iou = insec.sum() / union.sum()
        #                 if iou > 0.5:
        #                     num_tp_all += 1
        #                 else:
        #                     num_fp_all += 1

        #                 pred_inter_mask_gt_mov = pred_inter_mask.clone()
        #                 pred_inter_mask_gt_mov[~gt_mask] = 0
        #                 insec_gt_mask = (pred_inter_mask_gt_mov) & (gt_inter_mask)
        #                 union_gt_mask = (pred_inter_mask_gt_mov) | (gt_inter_mask)
        #                 iou_gt_mask = insec_gt_mask.sum() / union_gt_mask.sum()
        #                 if iou_gt_mask > 0.5:
        #                     num_tp_gt += 1
        #                 else:
        #                     num_fp_gt += 1
                            
        #                 pred_inter_mask_pred_mov = pred_inter_mask.clone()
        #                 pred_inter_mask_pred_mov[~pred_mov_mask] = 0
        #                 insec_pred_mask = (pred_inter_mask_pred_mov) & (gt_inter_mask)
        #                 union_pred_mask = (pred_inter_mask_pred_mov) | (gt_inter_mask)
        #                 iou_pred_mask = insec_pred_mask.sum() / union_pred_mask.sum()
        #                 if iou_pred_mask > 0.5:
        #                     num_tp_pred += 1
        #                 else:
        #                     num_fp_pred += 1
                            
        #                 print("    All mask size_gt: {}, size_pred: {}, intersection: {}, union: {}, iou: {}".format(
        #                     gt_inter_mask.sum(), (pred_inter_mask).sum(), insec.sum(), union.sum(), iou))
                        
        #                 print("        GT mask size_gt: {}, size_pred: {}, intersection: {}, union: {}, iou: {}".format( 
        #                     gt_inter_mask[gt_mask].sum(), (pred_inter_mask[gt_mask]).sum(), insec_gt_mask.sum(), union_gt_mask.sum(), iou_gt_mask))

        #                 print("        Pred mask size_gt: {}, size_pred: {}, intersection: {}, union: {}, iou: {}".format(
        #                     gt_inter_mask[pred_mov_mask].sum(), (pred_inter_mask[pred_mov_mask]).sum(), insec_pred_mask.sum(), union_pred_mask.sum(), iou_pred_mask))
        # if debug:
        #     if self.debug_counter % 20 == 0:
        #         print("num_tp: {}, num_fp: {}; num_tp_gt: {}, num_fp_gt: {}; num_tp_pred: {}, num_fp_pred: {}".format(
        #             num_tp_all, num_fp_all, num_tp_gt, num_fp_gt, num_tp_pred, num_fp_pred))
        #     self.debug_counter += 1

        return {
                "loss_ce_inter_mask": torch.sum(torch.stack(loss_masks)),
                "loss_dice_inter_mask": torch.sum(torch.stack(loss_dices)),
            }
            
    def loss_interaction_masks_inspect(self, outputs, targets, indices,
                               num_masks, mask_type, debug=False):
        loss_masks = []
        loss_dices = []
        loss_ce_scenes = []
        loss_dice_scenes = []
        for batch_id, (map_ids, target_ids) in enumerate(indices):
            num_masks = len(map_ids)
            pred_inter_vector= outputs["pred_interaction_dict"]['interaction_mask_vector'][batch_id] # [ num_points]
            pred_inter_mask = outputs["pred_interaction_dict"]['interaction_mask'][batch_id][:, map_ids] # [ num_points, num_masks]
            gt_inter_labels = targets[batch_id]['interaction_labels']
            gt_inter_mask = (gt_inter_labels != 0).float()
            gt_mov_masks = targets[batch_id]['masks'][target_ids, :] # [ num_masks, num_points]
            # got loss within each mov mask
            for idx in range(num_masks):
                gt_mov_mask = gt_mov_masks[idx]
                gt_mov_inter_mask = gt_inter_mask[gt_mov_mask].float().reshape(1, -1)
                pred_mov_inter_score = pred_inter_vector[gt_mov_mask].float().reshape(1, -1)
                loss_ce_inter_mask = sigmoid_ce_loss_jit(pred_mov_inter_score, gt_mov_inter_mask, 1)
                loss_dice_inter_mask = dice_loss_jit(pred_mov_inter_score, gt_mov_inter_mask, 1)
                loss_masks.append(loss_ce_inter_mask / num_masks)
                loss_dices.append(loss_dice_inter_mask/ num_masks)
            # got loss within each point
            loss_ce_scene = sigmoid_ce_loss_jit(pred_inter_vector.reshape(1,-1), gt_inter_mask.reshape(1,-1), 1)
            loss_dice_scene = dice_loss_jit(pred_inter_vector.reshape(1,-1), gt_inter_mask.reshape(1, -1), 1)
            loss_ce_scenes.append(loss_ce_scene)
            loss_dice_scenes.append(loss_dice_scene)
            
            # for debug use, print precision
            pred_mov_masks = outputs["pred_masks"][batch_id][:, map_ids].T # [ num_masks, num_points]
            pred_mov_masks = pred_mov_masks.sigmoid() > 0.5
            for idx in range(num_masks):
                gt_mov_mask = gt_mov_masks[idx]
                pred_mov_mask = pred_mov_masks[idx]
                pred_inter_mask = pred_inter_vector.sigmoid() > 0.5
                
                inter_mask_pr = pred_inter_mask[gt_mov_mask]
                inter_mask_gt = gt_inter_mask[gt_mov_mask]
                intersection = (pred_inter_mask_score.sigmoid() > 0.5) & (gt_inter_mask > 0.5)
                union = (pred_inter_mask_score.sigmoid() > 0.5) | (gt_inter_mask > 0.5)
                iou = intersection.sum() / union.sum()
                if iou > 0.5:
                    num_tp += 1
                else:
                    num_fp += 1
                print("   size_gt: {}, size_pred: {}, intersection: {}, union: {}, iou: {}".format(gt_inter_mask.sum(), (pred_inter_mask_score.sigmoid()>0.5).sum(), intersection.sum(), union.sum(), iou))
            
        loss_dict = {}
        if len(loss_masks) == 0:
            loss_dict = {
                "loss_ce_inter_mask": torch.tensor(0.0, device=pred_inter_mask.device),
                "loss_dice_inter_mask": torch.tensor(0.0, device=pred_inter_mask.device),
            }
        else:
            loss_dict =  {
                "loss_ce_inter_mask": torch.sum(torch.stack(loss_masks)),
                "loss_dice_inter_mask": torch.sum(torch.stack(loss_dices)),
            }
        loss_dict['loss_ce_inter_scene'] = torch.sum(torch.stack(loss_ce_scenes))
        loss_dict['loss_dice_inter_scene'] = torch.sum(torch.stack(loss_dice_scenes))
        return loss_dict
            
    def loss_interaction_centers(self, outputs, targets, indices,
                                    num_masks, mask_type, debug=False):
        loss_centers = []
        for batch_id, (map_ids, target_ids) in enumerate(indices):
            # print("batch_id: ", batch_id)
            # print("shape( outputs['pred_interaction_dict']['interaction_centers']): ", outputs["pred_interaction_dict"]["interaction_centers"][batch_id].shape)
            # print("map_ids: ", map_ids)
            pred_centers = outputs["pred_interaction_dict"]['interaction_centers'][batch_id][map_ids, :]
            gt_centers = targets[batch_id]['interaction_centers'][target_ids, :]
            if pred_centers.shape[0] == 0 or gt_centers.shape[0] == 0:
                loss_centers.append(torch.tensor(0.0, device=pred_centers.device))
                continue
            loss_centers.append(F.l1_loss(pred_centers, gt_centers, reduction='sum') / pred_centers.shape[0])
        return {
            "loss_interaction_centers": torch.sum(torch.stack(loss_centers)),
        }
    def loss_mov_part_centers(self, outputs, targets, indices,
                                    num_masks, mask_type, debug=False):
        loss_centers = []
        for batch_id, (map_ids, target_ids) in enumerate(indices):
            pred_centers = outputs["pred_mov_centers"][batch_id][map_ids, :]
            gt_centers = targets[batch_id]['mov_parts_centers'][target_ids, :]
            if pred_centers.shape[0] == 0 or gt_centers.shape[0] == 0:
                loss_centers.append(torch.tensor(0.0, device=pred_centers.device))
                continue
            loss_centers.append(F.l1_loss(pred_centers, gt_centers, reduction='sum') / pred_centers.shape[0])
        return {
            "loss_mov_part_centers": torch.sum(torch.stack(loss_centers)),
        }
            
    def loss_masks(self, outputs, targets, indices, 
                   num_masks, mask_type, debug=False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T # [ num_masks, num_points]
            target_mask = targets[batch_id][mask_type][target_id]

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        # del target_mask
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t[mask_type] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(
                point_logits, point_labels, num_masks, mask_type
            ),
            "loss_dice": dice_loss_jit(
                point_logits, point_labels, num_masks, mask_type
            ),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, 
                 num_masks, mask_type, debug = False):
        loss_map = {
            "labels": self.loss_labels, 
            "masks": self.loss_masks,
            "articulations": self.loss_articulations,
            "interaction_masks": self.loss_interaction_masks,
            "loss_interaction_masks_inspect": self.loss_interaction_masks_inspect,
            "interaction_centers": self.loss_interaction_centers,
            "loss_mov_part_centers": self.loss_mov_part_centers,}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type, debug)

    def forward(self, outputs, targets, mask_type, debug=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, num_masks, mask_type, debug
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                        mask_type,
                        debug
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
