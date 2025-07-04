import torch
import hydra
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP

from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr
from torch_geometric.nn import GATConv, GCNConv
from models.pointnet import PointNetfeat

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

class MultiGAT(nn.Module):
    def __init__(self, n_units=[17, 128, 100], n_heads=[2, 2], dropout=0.0):
        super(MultiGAT, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            in_channels = n_units[i] * n_heads[i-1] if i else n_units[i]
            layer_stack.append(GATConv(in_channels=in_channels, 
                                       out_channels=n_units[i+1], heads=n_heads[i]))
        
        self.layer_stack = nn.ModuleList(layer_stack)
    def forward(self, x, edges):
        
        for idx, gat_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, edges)
            if idx+1 < self.num_layers:
                x = F.elu(x)
        
        return x

class Connectivity3D(nn.Module):
    def __init__(
        self,
        config,
        gnn_channels,
        gnn_type,
        gat_heads,
        pt_out_dim=256,
        emb_dim = 256,
        connect_dim=256,
    ):
        super().__init__()

        self.config = config

        # backbone
        # self.backbone = hydra.utils.instantiate(config.backbone)
        # self.num_levels = len(self.hlevels)
        # sizes = self.backbone.PLANES[-5:]
        self.pt_out_dim, self.emb_dim = pt_out_dim, emb_dim
        self.object_encoder = PointNetfeat(global_feat=True, batch_norm=True, point_size=6, input_transform=False, feature_transform=False, out_size=self.pt_out_dim)
        self.object_embedding = nn.Linear(self.pt_out_dim, self.emb_dim)
        
        # part pointcloud pooling
        self.part_pooling = nn.ModuleList()
        ## first try identity pooling
        self.part_pooling.append(nn.Identity())
        
        # initialize graph neural network
        self.gnn_type = gnn_type
        self.gnn_channels = gnn_channels
        num_layers = len(gnn_channels) - 1
        self.gnn_layers = nn.ModuleList()
        if gnn_type == 'gat':
            self.gnn_layers = MultiGAT(n_units=gnn_channels, n_heads=gat_heads)
        elif gnn_type == 'gcn':
            self.gnn_layers = MultiGCN(n_units=gnn_channels)
        else:
            raise ValueError("Invalid gnn type: {}".format(gnn_type))
        
        # head for part-part connectivity
        self.connectivity_head = nn.Sequential(
            nn.Linear(emb_dim * 2, connect_dim),
            nn.ReLU(),
            nn.Linear(connect_dim, connect_dim),
            nn.ReLU(),
            nn.Linear(connect_dim, 1),
        )
       
    def forward(
            self,
            data
        ):
        # PointNet to get part-level embeddings
        part_points_batch = data['pcls_arr'] # [num_parts, num_points, 3]
        part_points_batch = part_points_batch.permute(0, 2, 1) # [num_parts, 3, num_points]
        points_emb = self.object_encoder(part_points_batch) # [num_parts, pt_out_dim]
        points_emb = self.object_embedding(points_emb) # [num_parts, emb_dim]
        
        # gnn to get part-part embeddings
        ## get edges for each object
        batch_size = data['batch_size']
        edges_batch = []
        num_edges = 0
        edge_matrix_idx_map_batch = []
        for bid in range(batch_size):
            edge_matrix_idx_map_list = []
            batch_start_idx = data['idxs_batch'][bid]
            batch_end_idx = data['idxs_batch'][bid+1]
            # parts_emb = points_emb[batch_start_idx:batch_end_idx]
            parts_idxs_list = data['parts_idxs_list_batch'][bid]
            num_objs = len(parts_idxs_list) - 1
            for obj_idx in range(num_objs):
                obj_start_idx = parts_idxs_list[obj_idx]
                obj_end_idx = parts_idxs_list[obj_idx+1]
                # obj_batch_emb = parts_emb[obj_start_idx:obj_end_idx] # [num_parts_in_obj, emb_dim]
                num_parts = obj_end_idx - obj_start_idx
                # gnn
                ## use fully connected graph within object for input
                edges_in = torch.tensor([[i+batch_start_idx, j+batch_start_idx] for i in range(num_parts) for j in range(num_parts) if i != j])
                edges_batch.append(edges_in)
                
                edge_matrix_idx_map = {}
                for i in range(num_parts):
                    for j in range(num_parts):
                        if i != j:
                            edge_matrix_idx_map[(i, j)] = num_edges
                            num_edges += 1
                edge_matrix_idx_map_list.append(edge_matrix_idx_map)
            edge_matrix_idx_map_batch.append(edge_matrix_idx_map_list)
                
        edges_batch = torch.cat(edges_batch, dim=0) # [num_edges, 2]
        edges_batch = edges_batch.T # [2, num_edges]
        ### to cuda
        edges_batch = edges_batch.to(points_emb.device)
        ## forward part embeddings to gnn
        if self.gnn_type == 'gat':
            part_embs = self.gnn_layers(points_emb, edges_batch)
        elif self.gnn_type == 'gcn':
            part_embs = self.gnn_layers(points_emb) # [num_parts, emb_dim]
        else:
            raise ValueError("Invalid gnn type: {}".format(self.gnn_type))
        
        ## concatenate part embeddings for each pair of parts, using edges_batch
        parent_part_embs = part_embs[edges_batch[0]]
        child_part_embs = part_embs[edges_batch[1]]
        part_pair_embs = torch.cat([parent_part_embs, child_part_embs], dim=1) # [num_edges, 2*emb_dim]
        connectivity = self.connectivity_head(part_pair_embs) # [num_edges, 1]
        connectivity = torch.tanh(connectivity) # [num_edges, 1] \in [-1, 1]
        
        # get connectivity within each object
        connectivity_batch = []
        for bid in range(batch_size):
            batch_start_idx = data['idxs_batch'][bid]
            batch_end_idx = data['idxs_batch'][bid+1]
            parts_idxs_list = data['parts_idxs_list_batch'][bid]
            num_objs = len(parts_idxs_list) - 1
            connectivity_list = []
            for obj_idx in range(num_objs):
                edge_matrix_idx_map = edge_matrix_idx_map_batch[bid][obj_idx]
                obj_start_idx = parts_idxs_list[obj_idx]
                obj_end_idx = parts_idxs_list[obj_idx+1]
                num_parts = obj_end_idx - obj_start_idx
                obj_connectivity = torch.zeros(num_parts, num_parts, device=connectivity.device, dtype=connectivity.dtype)
                # fill in the connectivity matrix
                for i in range(num_parts):
                    for j in range(num_parts):
                        if i != j:
                            edge_idx = edge_matrix_idx_map[(i, j)]
                            obj_connectivity[i, j] = connectivity[edge_idx]
                connectivity_list.append(obj_connectivity)
            connectivity_batch.append(connectivity_list)
        return connectivity_batch
                
        