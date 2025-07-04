import torch
import hydra
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch_geometric.nn import GATConv, GCNConv

from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr
from models.pointnet import PointNetfeat
from scipy.optimize import linear_sum_assignment

class MultiGCN(nn.Module):
    def __init__(self, n_units=[17, 128, 100], dropout=0.0):
        super(MultiGCN, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            layer_stack.append(GCNConv(in_channels=n_units[i], out_channels=n_units[i+1]))
        self.layer_stack = nn.ModuleList(layer_stack)
    
    def forward(self, x, edges):
        edges = edges.long()
        for idx, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x=x, edge_index=edges)
            if idx+1 < self.num_layers:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        indices = []

        # Iterate through batch size
        bs = len(targets)
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(
                -1
            )  # [num_queries, num_classes]
            tgt_ids = targets[b]["sem_ids"].clone()
            num_queries = out_prob.shape[0]
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][
                b
            ].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            if self.num_points != -1:
                point_idx = torch.randperm(
                    tgt_mask.shape[1], device=tgt_mask.device
                )[: int(self.num_points * tgt_mask.shape[1])]
                # point_idx = torch.randint(0, tgt_mask.shape[1], size=(self.num_points,), device=tgt_mask.device)
            else:
                # sample all points
                point_idx = torch.arange(
                    tgt_mask.shape[1], device=tgt_mask.device
                )

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class ArticulationGraph(nn.Module):
    def __init__(
        self,
        pcd_feats_size = 128,
        pcd_embed_size = 128,
        point_size = 3,
        coord_embed_size = 128,
        node_feat_size = 256,
        gnn_channels = [256, 128, 128],
        gat_heads = [2, 2],
        pt_out_dim=256,
        emb_dim = 256,
        connect_dim=256,
    ):
        super().__init__()
        
        # network
        self.coords_net = nn.ModuleList(
            nn.Conv1d(point_size, coord_embed_size, 1),
            nn.BatchNorm1d(coord_embed_size),
            nn.ReLU(),
            nn.Conv1d(coord_embed_size, coord_embed_size, 1),
            nn.BatchNorm1d(coord_embed_size),
            nn.ReLU()
        )
        self.pcd_features_net = nn.ModuleList(
            nn.Conv1d(pcd_feats_size, pcd_embed_size, 1),
            nn.BatchNorm1d(pcd_embed_size),
            nn.ReLU(),
            nn.Conv1d(pcd_embed_size, pcd_embed_size, 1),
            nn.BatchNorm1d(pcd_embed_size),
            nn.ReLU()
        )
        self.proj_map = nn.Linear(pcd_embed_size + coord_embed_size, node_feat_size)
        self.gnn_layers = MultiGCN(n_units=gnn_channels)
        
        self.connectivity_head = nn.Sequential(
            nn.Linear(node_feat_size * 2, connect_dim),   
            nn.ReLU(),
            nn.Linear(connect_dim, connect_dim),
            nn.ReLU(),
            nn.Linear(connect_dim, 1),
        ) 
            
        
        # matcher
        self.matcher = HungarianMatcher(
            cost_class=2,
            cost_mask=5,
            cost_dice=2,
            num_points=-1,
        )
        
    def forward(
            self,
            coords, 
            mov_pcd_feats, 
            mov_predictions,
            mov_targets,
            inter_pcd_feats,
            inter_predictions,
            inter_targets
        ):
        
        # match the moving and intermediate parts
        mov_match_indices = self.matcher(mov_predictions, mov_targets)
        inter_match_indices = self.matcher(inter_predictions, inter_targets)
        
        batch_size = len(mov_targets)
        
        node_emb_batch = []
        
        edges_batch = []
        mov_node_idx_start = 0
        inter_node_idx_start = 0
        
        num_gt_mov_nodes = []
        num_gt_inter_nodes = []
        num_matched_mov_nodes = []
        num_matched_inter_nodes = []
        
        gt_connectivity = []
        
        edge_matrix_idx_map_batch = []
        for bid in range(batch_size):
            # get the indices of the matched movable parts
            mov_match_indices_bid = mov_match_indices[bid]
            mov_matched_pred_idxs = mov_match_indices_bid[0]
            mov_matched_target_idxs = mov_match_indices_bid[1]
            assert len(mov_matched_pred_idxs) == len(mov_matched_target_idxs)
            # get embeds for the matched moving parts
            pred_masks_logits = mov_predictions["pred_masks"][bid]
            pred_masks_logits_matched = pred_masks_logits[:, mov_matched_pred_idxs]
            pred_masks_matched = pred_masks_logits_matched.sigmoid() > 0.5 # [num_points, num_matched_parts]
            
            for idx in range(pred_masks_matched.shape[1]):
                pred_mask_matched = pred_masks_matched[:, idx]
                part_coords = coords[bid, pred_mask_matched, :]
                part_feats = mov_pcd_feats[bid, pred_mask_matched, :]
                part_coords_emb = self.coords_net(part_coords.permute(1, 0)) # [emb_dim, num_points]
                part_feats_emb = self.pcd_features_net(part_feats.permute(1, 0)) # [emb_dim, num_points]
                part_coords_emb = torch.max(part_coords_emb, 1, keepdim=True)[0] # [emb_dim, 1]
                part_feats_emb = torch.max(part_feats_emb, 1, keepdim=True)[0] # [emb_dim, 1]
                
                node_feats = torch.cat([part_coords_emb, part_feats_emb], dim=0).permute(1, 0) # [1, node_feat_size]
                node_embs = self.proj_map(node_feats) # [1, node_feat_size]
                node_emb_batch.append(node_embs)
                
            # get the indices of the matched intermediate parts
            inter_match_indices_bid = inter_match_indices[bid]
            inter_matched_pred_idxs = inter_match_indices_bid[0]
            inter_matched_target_idxs = inter_match_indices_bid[1]
            assert len(inter_matched_pred_idxs) == len(inter_matched_target_idxs)
            # get embeds for the matched intermediate parts
            pred_masks_logits = inter_predictions["pred_masks"][bid]
            pred_masks_logits_matched = pred_masks_logits[:, inter_matched_pred_idxs]
            pred_masks_matched = pred_masks_logits_matched.sigmoid() > 0.5 # [num_points, num_matched_parts]
            for idx in range(pred_masks_matched.shape[1]):
                pred_mask_matched = pred_masks_matched[:, idx]
                part_coords = coords[bid, pred_mask_matched, :]
                part_feats = inter_pcd_feats[bid, pred_mask_matched, :]
                part_coords_emb = self.coords_net(part_coords.permute(1, 0))
                part_feats_emb = self.pcd_features_net(part_feats.permute(1, 0))
                part_coords_emb = torch.max(part_coords_emb, 1, keepdim=True)[0]
                part_feats_emb = torch.max(part_feats_emb, 1, keepdim=True)[0]
                
                node_feats = torch.cat([part_coords_emb, part_feats_emb], dim=0).permute(1, 0)
                node_embs = self.proj_map(node_feats)
                node_emb_batch.append(node_embs)
                
            # get edges for the moving parts and interactable parts
            mov_num_parts = len(mov_matched_pred_idxs)
            inter_num_parts = len(inter_matched_pred_idxs)
            inter_node_idx_start += mov_num_parts
            
            ## between moving parts
            for i in range(mov_num_parts):
                for j in range(mov_num_parts):
                    if i != j:
                        edges_batch.append(torch.tensor([mov_node_idx_start+i, mov_node_idx_start+j]))
            ## between moving and interactable parts
            for i in range(mov_num_parts):
                for j in range(inter_num_parts):
                    edges_batch.append(torch.tensor([mov_node_idx_start+i, inter_node_idx_start+j]))
            ## between interactable parts
            for i in range(inter_num_parts):
                for j in range(inter_num_parts):
                    if i != j:
                        edges_batch.append(torch.tensor([inter_node_idx_start+i, inter_node_idx_start+j]))
                        
            mov_node_idx_start += mov_num_parts
            inter_node_idx_start += inter_num_parts
            
            num_gt_mov_nodes.append(len(mov_targets[bid]["inst_ids"]))
            num_gt_inter_nodes.append(len(inter_targets[bid]["inst_ids"]))
            num_matched_mov_nodes.append(mov_num_parts)
            num_matched_inter_nodes.append(inter_num_parts)
            
            # get ground truth connectivity
            mov_inst_ids = mov_targets[bid]["inst_ids"]
            inter_inst_ids = inter_targets[bid]["inst_ids"]
            mov_inst_ids_matched = mov_inst_ids[mov_matched_target_idxs]
            inter_inst_ids_matched = inter_inst_ids[inter_matched_target_idxs]
            connectivity = torch.zeros((mov_num_parts, inter_num_parts))
            # if inst_id equals and is not zero, then there is a connection:
            for i in range(mov_num_parts):
                for j in range(inter_num_parts):
                    if mov_inst_ids_matched[i] == inter_inst_ids_matched[j] and mov_inst_ids_matched[i] != 0:
                        connectivity[i, j] = 1
            gt_connectivity.append(connectivity)
            
        node_embs_batch = torch.cat(node_emb_batch, dim=0) # [num_nodes, node_feat_size]
        edges_batch = torch.cat(edges_batch, dim=0)
            
        node_embs_gnn = self.gnn_layers(node_embs_batch, edges_batch)
        
        # predict connectivity
        pred_connectivity = []
        mov_start_idx = 0
        inter_start_idx = 0
        for bid in range(batch_size):
            inter_start_idx += num_matched_mov_nodes[bid]
            
            mov_num_parts = num_matched_mov_nodes[bid]
            inter_num_parts = num_matched_inter_nodes[bid]
            
            mov_node_embs = node_embs_gnn[mov_start_idx:mov_start_idx+mov_num_parts]
            inter_node_embs = node_embs_gnn[inter_start_idx:inter_start_idx+inter_num_parts]
            
            # concatenate node embeddings for each pair of mov and inter parts
            paired_embs = []
            for i in range(mov_num_parts):
                for j in range(inter_num_parts):
                    paired_embs.append(torch.cat([mov_node_embs[i], inter_node_embs[j]], dim=0))
            paired_embs = torch.stack(paired_embs, dim=0) # [ num_mov_parts * num_inter_parts, 2*node_feat_size]
            connectivity = self.connectivity_head(paired_embs)
            connectivity = torch.sigmoid(connectivity)
            connectivity = connectivity.reshape(mov_num_parts, inter_num_parts)
            pred_connectivity.append(connectivity)
            
            mov_start_idx += mov_num_parts
            inter_start_idx += inter_num_parts
            
        return {
            'gt_connectivity': gt_connectivity,
            'pred_connectivity': pred_connectivity,
            'num_gt_mov_nodes': num_gt_mov_nodes,
            'num_gt_inter_nodes': num_gt_inter_nodes,
            'num_matched_mov_nodes': num_matched_mov_nodes,
            'num_matched_inter_nodes': num_matched_inter_nodes
        }
            
            
            
    