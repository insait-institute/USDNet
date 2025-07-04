import torch
import hydra
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr

class Mask3D(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
        predict_articulation = False,
        predict_articulation_mode = 0,
        articulation_type_mapping = None,
        predict_hierarchy_interaction = False,
        predict_hierarchy_interaction_mode = 0,
        predict_interaction_centers = False,
        use_interaction_queries = False,
        mov_inter_couple = False,
        reserve_arti_net = False,
        pred_mov_centers = False,
        pointwise_origin = True,
        pointwise_axis = True,
    ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        self.predict_articulation = predict_articulation
        self.predict_articulation_mode = predict_articulation_mode
        self.articulation_type_mapping = articulation_type_mapping
        self.predict_hierarchy_interaction = predict_hierarchy_interaction
        self.predict_hierarchy_interaction_mode = predict_hierarchy_interaction_mode
        self.predict_interaction_centers = predict_interaction_centers
        self.use_interaction_queries = use_interaction_queries
        self.mov_inter_couple= mov_inter_couple
        self.reserve_arti_net = reserve_arti_net
        self.pred_mov_centers = pred_mov_centers
        self.pointwise_origin = pointwise_origin
        self.pointwise_axis = pointwise_axis
        
        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]

        self.mask_features_head = conv(
            self.backbone.PLANES[7],
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )
        if self.use_interaction_queries:
            # set interaction queries as learnable parameters, size(num_queries, D)
            self.interaction_backbone = hydra.utils.instantiate(config.backbone)
            # self.interaction_queries = nn.Parameter(torch.randn(self.num_queries, self.mask_dim))
            self.mov_to_interaction_queries = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            self.interaction_queries_res = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            
            self.interaction_query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
            
        #     self.interaction_decoder_norm = nn.LayerNorm(hidden_dim)
            
            self.interaction_cross_attention = nn.ModuleList()
            self.interaction_self_attention = nn.ModuleList()
            self.interaction_ffn_attention = nn.ModuleList()
            self.interaction_lin_squeeze = nn.ModuleList()

            num_shared = self.num_decoders if not self.shared_decoder else 1
            for _ in range(num_shared):
                tmp_cross_attention = nn.ModuleList()
                tmp_self_attention = nn.ModuleList()
                tmp_ffn_attention = nn.ModuleList()
                tmp_squeeze_attention = nn.ModuleList()
                for i, hlevel in enumerate(self.hlevels):
                    tmp_cross_attention.append(
                        CrossAttentionLayer(
                            d_model=self.mask_dim,
                            nhead=self.num_heads,
                            dropout=self.dropout,
                            normalize_before=self.pre_norm,
                        )
                    )

                    tmp_squeeze_attention.append(
                        nn.Linear(sizes[hlevel], self.mask_dim)
                    )
                    tmp_self_attention.append(
                        SelfAttentionLayer(
                            d_model=self.mask_dim,
                            nhead=self.num_heads,
                            dropout=self.dropout,
                            normalize_before=self.pre_norm,
                        )
                    )

                    tmp_ffn_attention.append(
                        FFNLayer(
                            d_model=self.mask_dim,
                            dim_feedforward=dim_feedforward,
                            dropout=self.dropout,
                            normalize_before=self.pre_norm,
                        )
                    )

                self.interaction_cross_attention.append(tmp_cross_attention)
                self.interaction_self_attention.append(tmp_self_attention)
                self.interaction_ffn_attention.append(tmp_ffn_attention)
                self.interaction_lin_squeeze.append(tmp_squeeze_attention)
    

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(
                mask, p2s, dim=dim
            )[0]
        else:
            assert False, "Scatter function not known"

        assert (
            not use_np_features
        ) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(self.num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes),
            )
        if self.predict_articulation or self.reserve_arti_net:
        # if True:
            if self.predict_articulation_mode == 0:
                self.origin_embed_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 3),
                    )
                self.axis_embed_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                )
            elif self.predict_articulation_mode == 1 or \
                self.predict_articulation_mode == 2 or \
                self.predict_articulation_mode == 3 or \
                self.predict_articulation_mode == 4:
                self.origin_embed_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 3),
                    )
                self.axis_embed_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                )
                self.axis_embed_head_query = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                )
                self.feature_arti_emb = conv(
                    self.backbone.PLANES[7],
                    self.mask_dim,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    D=3,
                )
                
        if self.pred_mov_centers:
            self.mov_center_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
                
        if self.predict_hierarchy_interaction:
            self.initialize_interaction_nn(hidden_dim)

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
            )
        else:
            assert False, "pos enc type not known"

        self.pooling = MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=3
        )

        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_squeeze_attention.append(
                    nn.Linear(sizes[hlevel], self.mask_dim)
                )

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm,
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
            

    def initialize_interaction_nn(self, hidden_dim):       
        self.feature_interaction_emb_conv = conv(
            self.backbone.PLANES[7],
            self.mask_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            D=3,
        )
        self.feature_interaction_emb_norm = get_norm(
            NormType.BATCH_NORM, self.mask_dim, 3, bn_momentum=0.1
        )
        self.feature_interaction_emb_relu = MinkowskiReLU(inplace=True)
        self.mask_interaction_emb = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
        if self.predict_hierarchy_interaction_mode == 1 or self.predict_hierarchy_interaction_mode == 2:
            self.mask_interaction_emb_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        if self.predict_interaction_centers:
            self.interaction_center_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
            )
    def interaction_emb_nn(self, pcd_features):
        interaction_features = self.feature_interaction_emb_conv(pcd_features)
        interaction_features = self.feature_interaction_emb_norm(interaction_features)
        interaction_features = self.feature_interaction_emb_relu(interaction_features)
        return interaction_features

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd
    
    def articulation_forward(self, queries, outputs_mask, 
                             arti_features=None, 
                             output_class=None):
        outputs_origins, outputs_axises = None, None
        if self.predict_articulation_mode == 0: 
            # predict origin and axis from queries, naively
            outputs_axises = self.axis_embed_head(queries)
            outputs_origins = self.origin_embed_head(queries)
            # print("         outputs_origins.shape: {}".format(outputs_origins.shape))
            # print("         outputs_axises.shape: {}".format(outputs_axises.shape))
        elif self.predict_articulation_mode == 1: 
            # predict origin and axis from masked features, origin prediciton in the global coordinate
            arti_features = torch.stack(arti_features.decomposed_features)
            feature_masks = torch.stack(outputs_mask)
            feature_masks = feature_masks.sigmoid() > 0.5
            batch_size, num_points, num_queries = feature_masks.shape
            outputs_origins_list = []
            outputs_axises_list = []
            # print(" num of points: {}".format(num_points))
            for bid in range(batch_size):
                outputs_origins_bid_list = []
                outputs_axises_bid_list = []
                for qid in range(num_queries):
                    feature_mask = feature_masks[bid, :, qid]
                    query_arti_feature = arti_features[bid, feature_mask, :]
                    if query_arti_feature.shape[0] == 0:
                        predict_origin = torch.zeros(3, device=queries.device)
                        predict_axis = torch.zeros(3, device=queries.device)
                    else:
                        predict_origin = self.origin_embed_head(query_arti_feature)
                        predict_axis = self.axis_embed_head(query_arti_feature)
                        predict_origin = predict_origin.mean(dim=0)
                        predict_axis = predict_axis.mean(dim=0)
                    outputs_origins_bid_list.append(predict_origin)
                    outputs_axises_bid_list.append(predict_axis)
                outputs_origins_list.append(torch.stack(outputs_origins_bid_list))
                outputs_axises_list.append(torch.stack(outputs_axises_bid_list))
            outputs_origins = torch.stack(outputs_origins_list)
            outputs_axises = torch.stack(outputs_axises_list)
        elif self.predict_articulation_mode == 2:
            # predict origin and axis from masked features , origin prediciton in the local coordinate
            # predict origin and axis from masked features
            if self.pointwise_axis or self.pointwise_origin:
                coordinates = torch.stack(arti_features.decomposed_coordinates) * self.voxel_size # (batch_size, point_num, 3)
                coordinates_expanded = coordinates.unsqueeze(2) # (batch_size, point_num, 1, 3)
                arti_features = torch.stack(arti_features.decomposed_features) # (batch_size, point_num, D)
                feature_masks = torch.stack(outputs_mask) # (batch_size, point_num, num_query)
                feature_masks = feature_masks.sigmoid() > 0.5
                mask_expanded = feature_masks.unsqueeze(-1) # (batch_size, point_num, num_query, 1)
                num_masked_points = mask_expanded.sum(dim=1) # (batch_size, num_query, 1)
                batch_size, num_points, num_queries = feature_masks.shape

            if self.pointwise_origin:
                origin_features = self.origin_embed_head(arti_features) # (batch_size, point_num, D_hidden)
                origin_features= origin_features.unsqueeze(2) # (batch_size, point_num, 1, 3)
                origin_features = origin_features.expand(-1, -1, num_queries, -1) # (batch_size, point_num, num_queries, 3)
                masked_origin_features = (origin_features + coordinates_expanded) * mask_expanded # (batch_size, point_num, num_query, 3)
                outputs_origins = masked_origin_features.sum(dim=1) / ( num_masked_points + 1e-6) # (batch_size, num_query, 3)
                num_masked_points_sq = num_masked_points.squeeze(-1) # (batch_size, num_query)
                outputs_origins[num_masked_points_sq < 0.5] = torch.zeros(3, device=queries.device).float()
            else:
                outputs_origins = self.origin_embed_head(queries) # (batch_size, num_query, 3)
                
            # set origin and axis of queries without any points to zero
            if self.pointwise_axis:
                axis_features = self.axis_embed_head(arti_features) # (batch_size, point_num, D_hidden)
                axis_features = axis_features.unsqueeze(2) # (batch_size, point_num, 1, 3)
                axis_features = axis_features.expand(-1, -1, num_queries, -1) # (batch_size, point_num, num_queries, 3)
                masked_axis_features = axis_features * mask_expanded # (batch_size, point_num, num_query, 3)
                num_masked_points_sq = num_masked_points.squeeze(-1) # (batch_size, num_query)
                # calculate mean of origin and axis, masked by feature_masks
                outputs_axises = masked_axis_features.sum(dim=1) / ( num_masked_points + 1e-6) \
                    + self.axis_embed_head_query(queries) # (batch_size, num_query, 3) 
                outputs_axises[num_masked_points_sq < 0.5] = torch.tensor([0, 0, 1], device=queries.device).float()  
            else:
                outputs_axises = self.axis_embed_head(queries)
            
            # batch_size, num_queries = queries.shape[:2]
            # outputs_origins= torch.zeros((batch_size, num_queries, 3), device=queries.device)
            # outputs_axises = torch.zeros((batch_size, num_queries, 3), device=queries.device)
            
        elif self.predict_articulation_mode == 3:
            # # predict origin and axis heuristically from masked point clouds, center as origin and 
            # # eigen vector as axis
            coordinates = arti_features.decomposed_coordinates
            coordinates = torch.stack(coordinates)
            feature_masks = torch.stack(outputs_mask)
            feature_masks = feature_masks.sigmoid() > 0.5
            # use heuristic articulation prediction 
            pred_logits = output_class
            pred_classes = torch.functional.F.softmax(pred_logits, dim = -1)
            outputs_origins_list = []
            outputs_axises_list = []
            for bid in range(pred_logits.shape[0]):
                outputs_origins_bid_list = []
                outputs_axises_bid_list = []
                pred_logits_batch = pred_logits[bid]
                # get class with max logits
                pred_classes = torch.argmax(pred_logits_batch, dim=-1)
                for qid in range(pred_classes.shape[0]):
                    pred_class = pred_classes[qid]
                    if pred_class == self.articulation_type_mapping['rotation'] or \
                        pred_class == self.articulation_type_mapping['translation']:
                        feature_mask = feature_masks[bid, :, qid]
                        coords = coordinates[bid, feature_mask, :] * self.voxel_size
                        if coords.shape[0] < 10:
                            predict_origin = torch.zeros(3, device=queries.device)
                            predict_axis = torch.zeros(3, device=queries.device)
                            outputs_origins_bid_list.append(predict_origin)
                            outputs_axises_bid_list.append(predict_axis)
                        else:
                            mean_point = coords.mean(dim=0)
                            centered_pc = coords - mean_point
                            cov_matrix = torch.mm(centered_pc.T, centered_pc) / (centered_pc.shape[0] - 1)
                            eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
                            eigenvalues = eigenvalues.real
                            eigenvectors = eigenvectors.real
                            sorted_indices = torch.argsort(eigenvalues, descending=True)
                            principal_directions = eigenvectors[:, sorted_indices]  # These are the 3 PCA axes (directions)
                            # to rows 
                            principal_directions = principal_directions.T
                            outputs_origins_bid_list.append(mean_point.float())
                            # for rotation, axis is the eigen vector of the largest eigen value
                            # for translation, axis is the eigen vector of the smallest eigen value
                            if pred_class == self.articulation_type_mapping['rotation']:
                                outputs_axises_bid_list.append(principal_directions[0].float())
                            else:
                                outputs_axises_bid_list.append(principal_directions[2].float())
                    else:
                        outputs_origins_bid_list.append(torch.zeros(3, device=queries.device))
                        # set direction to [0, 0, 1]
                        outputs_axises_bid_list.append( torch.tensor([0, 0, 1], device=queries.device))  
                outputs_origins_list.append(torch.stack(outputs_origins_bid_list))
                outputs_axises_list.append(torch.stack(outputs_axises_bid_list))
            outputs_origins = torch.stack(outputs_origins_list).float()
            outputs_axises = torch.stack(outputs_axises_list).float()
        elif self.predict_articulation_mode == 4:
            coordinates = torch.stack(arti_features.decomposed_coordinates) * self.voxel_size # (batch_size, point_num, 3)
            arti_features = torch.stack(arti_features.decomposed_features) # (batch_size, point_num, D)
            feature_masks = torch.stack(outputs_mask) # (batch_size, point_num, num_query)
            feature_masks = feature_masks.sigmoid() > 0.5
            batch_size, num_points, num_queries = feature_masks.shape
            origin_features = self.origin_embed_head(arti_features) # (batch_size, point_num, D_hidden)
            axis_features = self.axis_embed_head(arti_features) # (batch_size, point_num, D_hidden)
            # Default eigenvector for masks with no points
            default_eigenvectors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                                                device=queries.device).float().unsqueeze(0).unsqueeze(0).expand(batch_size, num_queries, 3, 3)

            # Mask out non-relevant points by setting points outside the mask to NaN (ignored in mean/cov calculations)
            # print("    feature_masks.unsqueeze(-1).shape: {}".format(feature_masks.unsqueeze(-1).shape))
            # print("    coordinates.unsqueeze(2).shape: {}".format(coordinates.unsqueeze(2).shape))
            masked_coordinates = torch.where(feature_masks.unsqueeze(-1), coordinates.unsqueeze(2),
                    torch.tensor(float('nan'), device=coordinates.device)) # (batch_size, num_points, num_queries, 3)
            # Calculate the mean, ignoring NaNs
            mean_coordinates = torch.nanmean(masked_coordinates, dim=1, keepdim=True)  # (batch_size, 1, num_queries, 3)
            # print("    masked_coordinates.shape: {}".format(masked_coordinates.shape))
            # print("    mean_coordinates.shape: {}".format(mean_coordinates.shape))
            # Center points by subtracting the mean, while ignoring NaNs
            centered_coordinates = masked_coordinates - mean_coordinates  # (batch_size, num_points, num_queries, 3)
            # Replace NaNs with zero after centering to allow matrix operations
            centered_coordinates = torch.nan_to_num(centered_coordinates, nan=0.0)
            # swith to (batch_size, num_queries, num_points, 3)
            centered_coordinates = centered_coordinates.permute(0, 2, 1, 3)
            # centered_coordinates_T = centered_coordinates.permute(0, 1, 3, 2) # (batch_size, num_queries, 3, num_points)
            # Compute covariance matrix, taking only masked points into account
            num_masked_points = feature_masks.sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1, num_queries)
            # print("    centered_coordinates.shape: {}".format(centered_coordinates.shape))
            # print("    centered_coordinates_T.shape: {}".format(centered_coordinates_T.shape))
            # cov_matrices = torch.einsum('bqpi, bqpj -> bqij', centered_coordinates, centered_coordinates) / (num_masked_points - 1)
            # Reshape centered_coordinates to (batch_size * num_queries, num_points, 3)
            reshaped_coords = centered_coordinates.view(batch_size * num_queries, num_points, 3)
            # Compute the covariance matrices: (B*Q, 3, num_points) x (B*Q, num_points, 3) -> (B*Q, 3, 3)
            cov_matrices = torch.bmm(reshaped_coords.transpose(1, 2), reshaped_coords) / (num_masked_points.view(-1, 1, 1) - 1)
            # Reshape back to (batch_size, num_queries, 3, 3)
            cov_matrices = cov_matrices.view(batch_size, num_queries, 3, 3)

            # Perform SVD to get principal components
            U, S, Vh = torch.linalg.svd(cov_matrices)  # Vh has shape (batch_size, num_queries, 3, 3)
            # Assign default eigenvectors for masks with no points, eigenvectors in rows
            eigen_vector_mask = num_masked_points.squeeze(1) > 0.5 # (batch_size, num_queries)
            eigen_vector_mask = eigen_vector_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_queries, 3, 3)
            eigenvectors = torch.where(eigen_vector_mask, Vh, default_eigenvectors) # (batch_size, num_queries, 3, 3)
            
            # predict origin and axis from masked features
            origin_features= origin_features.unsqueeze(2) # (batch_size, point_num, 1, 3)
            origin_features = origin_features.expand(-1, -1, num_queries, -1) # (batch_size, point_num, num_queries, 3)
            axis_features = axis_features.unsqueeze(2) # (batch_size, point_num, 1, 3)
            axis_features = axis_features.expand(-1, -1, num_queries, -1) # (batch_size, point_num, num_queries, 3)
            coordinates_expanded = coordinates.unsqueeze(2) # (batch_size, point_num, 1, 3)
            mask_expanded = feature_masks.unsqueeze(-1) # (batch_size, point_num, num_query, 1)
            masked_origin_features = (origin_features + coordinates_expanded) * mask_expanded # (batch_size, point_num, num_query, 3)
            masked_axis_features = axis_features * mask_expanded # (batch_size, point_num, num_query, 3)
            num_masked_points = mask_expanded.sum(dim=1) # (batch_size, num_query, 1)
            # calculate mean of origin and axis, masked by feature_masks
            outputs_origins = masked_origin_features.sum(dim=1) / ( num_masked_points + 1e-6) # (batch_size, num_query, 3)
            outputs_axises = masked_axis_features.sum(dim=1) / ( num_masked_points + 1e-6) # (batch_size, num_query, 3)
            ## transfrom axis from local pca to global coordinate
            outputs_axises = torch.einsum('bqk,bqkl->bql', outputs_axises, eigenvectors) # (batch_size, num_query, 3)
            
            # set origin and axis of queries without any points to zero
            num_masked_points = num_masked_points.squeeze(-1) # (batch_size, num_query)
            outputs_origins[num_masked_points < 0.5] = torch.zeros(3, device=queries.device).float()
            outputs_axises[num_masked_points < 0.5] = torch.tensor([0, 0, 1], device=queries.device).float() 
            
        return outputs_origins, outputs_axises
    def hierarchy_interaction_forward(self, queries, outputs_mask,
            interaction_features):
        centers_list = []
        if self.predict_hierarchy_interaction_mode == 0:
            # use pcd features directly get heatmap
            interaction_features = torch.stack(interaction_features.decomposed_features) # (batch_size, point_num, D)
            interaction_mask_features = self.mask_interaction_emb(interaction_features) # (batch_size, point_num, 1)
            interaction_masks = interaction_mask_features.sigmoid()
        elif self.predict_hierarchy_interaction_mode == 1:
            interaction_masks = []
            interaction_masks_out = []
            interaction_outs = []
            interaction_masks_vecotr = []
            # pcd query to do cross attention with masked features
            batch_size = len(interaction_features.decomposed_features)
            
            for bid in range(batch_size):
                interaction_features_bid = interaction_features.decomposed_features[bid] # (point_num, D)
                query_emb = self.mask_interaction_emb_head(queries[bid]) # ( num_query, D)
                interaction_mask = interaction_features_bid @ query_emb.T # (point_num, num_query)
                mov_masks = outputs_mask[bid].sigmoid()>0.5 # (point_num, num_query)
                # interaction_mask = interaction_mask * mov_masks # (point_num, num_query)
                
                interaction_mask_out = (interaction_mask.sigmoid()>0.5) & mov_masks
                interaction_out = interaction_mask.sigmoid()>0.5
                interaction_mask_vecotr = interaction_mask_out.max(dim = 1, keepdim = True)[0] # (point_num, 1)
                interaction_masks_vecotr.append( interaction_mask_vecotr) # (point_num, 1)
                interaction_outs.append( interaction_out) # (point_num, num_query)
                interaction_masks_out.append( interaction_mask_out) # (point_num, num_query)
                interaction_masks.append( interaction_mask) # (point_num, num_query)
                if self.predict_interaction_centers:
                    num_queries = queries[bid].shape[0]
                    interaction_features_bid = interaction_features.decomposed_features[bid] # (point_num, D)
                    coordinates = interaction_features.decomposed_coordinates[bid] * self.voxel_size # (point_num, 3)
                    pred_interaction_centers = self.interaction_center_head(interaction_features_bid) + coordinates # (point_num, 3)
                    pred_interaction_centers_expand = pred_interaction_centers.unsqueeze(1).expand(-1, num_queries, -1) # (point_num, num_query, 3)
                    mask_expanded = mov_masks.unsqueeze(-1).expand(-1, -1, 3) # (point_num, num_query, 3)
                    masked_pred_interaction_centers = pred_interaction_centers_expand * mask_expanded # (point_num, num_query, 3)
                    out_pred_interaction_centers = masked_pred_interaction_centers.sum(dim=0) / (mov_masks.sum(dim=0).unsqueeze(-1) + 1e-6) # (num_query, 3)
                    centers_list.append(out_pred_interaction_centers)  
        else:
            raise NotImplementedError 
        return {
            "interaction_mask_vector": interaction_masks_vecotr,
            'interaction_mask_out': interaction_masks_out,
            "interaction_outs": interaction_outs,
            "interaction_mask": interaction_masks,
            "interaction_centers": centers_list
        }
        
    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False,
        matcher=None, use_gt_movable_mask = False, target=None,
        use_gt_movable_mask_as_pred = False
    ):
        pcd_features, aux = self.backbone(x)
        # print("aux type: {}".format(type(aux)))
        if self.use_interaction_queries:
            pcd_interaction_features, aux_interaction = self.interaction_backbone(x)
            # print("aux_interaction type: {}".format(type(aux_interaction)))
        batch_size = len(x.decomposed_coordinates)

        pcd_coords = pcd_features.decomposed_coordinates
        pcd_coords = torch.stack(pcd_coords)

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates,
                coordinate_manager=aux[-1].coordinate_manager,
                coordinate_map_key=aux[-1].coordinate_map_key,
                device=aux[-1].device,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)
        arti_features = None
        if self.predict_articulation and \
            (self.predict_articulation_mode == 1 or \
             self.predict_articulation_mode == 2 or \
             self.predict_articulation_mode == 3 or \
             self.predict_articulation_mode == 4):
            arti_features = self.feature_arti_emb(pcd_features)
        if self.predict_hierarchy_interaction:
            interaction_features = self.interaction_emb_nn(pcd_interaction_features)

        if self.train_on_segments:
            mask_segments = []
            for i, mask_feature in enumerate(
                mask_features.decomposed_features
            ):
                mask_segments.append(
                    self.scatter_fn(mask_feature, point2segment[i], dim=0)
                )

        sampled_coords = None

        if self.non_parametric_queries:
            fps_idx = [
                furthest_point_sample(
                    x.decomposed_coordinates[i][None, ...].float(),
                    self.num_queries,
                )
                .squeeze(0)
                .long()
                for i in range(len(x.decomposed_coordinates))
            ]

            sampled_coords = torch.stack(
                [
                    coordinates.decomposed_features[i][fps_idx[i].long(), :]
                    for i in range(len(fps_idx))
                ] # batch_size, num_queries, D
            )

            mins = torch.stack(
                [
                    coordinates.decomposed_features[i].min(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )
            maxs = torch.stack(
                [
                    coordinates.decomposed_features[i].max(dim=0)[0]
                    for i in range(len(coordinates.decomposed_features))
                ]
            )

            query_pos = self.pos_enc(
                sampled_coords.float(), input_range=[mins, maxs]
            )  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)
            if not self.use_np_features:
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = torch.stack(
                    [
                        pcd_features.decomposed_features[i][
                            fps_idx[i].long(), :
                        ]
                        for i in range(len(fps_idx))
                    ]
                )
                queries = self.np_feature_projection(queries)
            query_pos = query_pos.permute((2, 0, 1))

        elif self.random_queries:
            query_pos = (
                torch.rand(
                    batch_size,
                    self.mask_dim,
                    self.num_queries,
                    device=x.device,
                )
                - 0.5
            )

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = (
                    torch.rand(
                        batch_size,
                        2 * self.mask_dim,
                        self.num_queries,
                        device=x.device,
                    )
                    - 0.5
                )
            else:
                query_pos_feat = torch.randn(
                    batch_size,
                    2 * self.mask_dim,
                    self.num_queries,
                    device=x.device,
                )

            queries = query_pos_feat[:, : self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim :, :].permute(
                (2, 0, 1)
            )
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(
                1, batch_size, 1
            )

        predictions_class = []
        predictions_mask = []
        pred_mov_centers_list = []
        predictions_origins = []
        predictions_axises = []
        predictions_interaction_dicts = []
        
        if self.use_interaction_queries:
            # queries_for_interaction = interaction_queries
            # queries_for_interaction = self.interaction_queries.unsqueeze(0).repeat(batch_size, 1, 1)

            queries_for_interaction = torch.zeros_like(queries)
            interaction_query_pos = query_pos

        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                if self.train_on_segments:
                    output_class, outputs_mask, attn_mask = self.mask_module(
                        queries,
                        mask_features,
                        mask_segments,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=point2segment,
                        coords=coords,
                    )
                else:
                    output_class, outputs_mask, attn_mask = self.mask_module(
                        queries,
                        mask_features,
                        None,
                        len(aux) - hlevel - 1,
                        ret_attn_mask=True,
                        point2segment=None,
                        coords=coords,
                    )
                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features
                
                if self.pred_mov_centers:
                    pred_mov_centers_out_batch = []
                    for bid in range(batch_size):
                        coordinates = mask_features.decomposed_coordinates[bid] * self.voxel_size # (point_num, 3)
                        pred_mov_centers = self.mov_center_head(mask_features.decomposed_features[bid]) + coordinates # (num_points, 3) 
                        pred_mov_centers_expand = pred_mov_centers.unsqueeze(1).expand(-1, self.num_queries, -1) # (num_points, num_queries, 3)
                        
                        mov_mask_expanded = (outputs_mask[bid].sigmoid()>0.5).unsqueeze(-1).expand(-1, -1, 3) # (num_points, num_queries, 3)
                        pred_mov_centers_out = mov_mask_expanded * pred_mov_centers_expand
                        # print("pred_mov_centers_out.shape: {}".format(pred_mov_centers_out.shape))
                        # print("mov_mask_expanded.shape: {}".format(mov_mask_expanded.shape))
                        pred_mov_centers_out = pred_mov_centers_out.sum(dim=0) / (mov_mask_expanded.sum(dim=0) + 1e-6) # (num_query, 3)
                        pred_mov_centers_out_batch.append(pred_mov_centers_out)
                    pred_mov_centers_list.append(pred_mov_centers_out_batch)
                if self.predict_articulation:
                    outputs_origins, outputs_axises = \
                        self.articulation_forward(queries, outputs_mask, 
                                                  arti_features = arti_features,
                                                  output_class = output_class)
                        
                    predictions_origins.append(outputs_origins)
                    predictions_axises.append(outputs_axises)
                if self.predict_hierarchy_interaction:
                    if not self.use_interaction_queries:
                        queries_for_interaction = queries

                    if use_gt_movable_mask:
                        # use ground truth movable mask for interaction prediction
                        outputs_level = {
                            "pred_logits": output_class,
                            "pred_masks": outputs_mask,
                        }
                        indices = matcher(outputs_level, target, "masks")
                        # for batch_id, (map_ids, target_ids) in enumerate(indices):
                        #     pred_mov_masks_matched = 
                        queries_matched = []
                        gt_masks_matched = []
                        output_classes_matched = []
                        output_classes_gt = []
                        output_maskes_gt= []
                        for batch_id, (map_ids, target_ids) in enumerate(indices):
                            queries_matched.append(queries_for_interaction[batch_id][map_ids])
                            gt_masks_matched.append(target[batch_id]["masks"][target_ids].T)
                            output_classes_matched.append(output_class[batch_id][map_ids])
                            if use_gt_movable_mask_as_pred:
                                output_maskes_gt_bid = torch.zeros_like(outputs_mask[batch_id]) # (num_points, num_queries)
                                output_maskes_gt_bid[:, map_ids] = target[batch_id]["masks"][target_ids].T.float()
                                output_maskes_gt.append(output_maskes_gt_bid)
                                # num_queries = self.num_queries
                                # gt_masks = target[batch_id]["masks"].T
                                # output_maskes_gt_bid = torch.ones(gt_masks.shape[0], gt_masks.shape[1], device=queries.device).float()
                                # output_maskes_gt_bid[gt_masks] = 1
                                # output_maskes_gt_bid[~gt_masks] = -1
                                # pad = torch.ones(gt_masks.shape[0], num_queries - gt_masks.shape[1], device=queries.device).float() * -1.0
                                # output_maskes_gt_bid = torch.cat([output_maskes_gt_bid, pad], dim=1)
                                # output_maskes_gt.append(output_maskes_gt_bid)   
                        if use_gt_movable_mask_as_pred:
                            outputs_mask = output_maskes_gt
                        interaction_dict_matched = self.hierarchy_interaction_forward(queries_matched, gt_masks_matched, interaction_features)
                        # fill interaction_dict with full num_queries
                        interaction_mask_matched = interaction_dict_matched["interaction_mask"]
                        interaction_centers_matched = interaction_dict_matched["interaction_centers"]
                        interaction_mask_list = []
                        interaction_centers_list = []
                        for batch_id, (map_ids, target_ids) in enumerate(indices):
                            num_points = interaction_mask_matched[batch_id].shape[0]
                            interaction_mask = torch.zeros( num_points, self.num_queries, device=queries.device)
                            interaction_center = torch.zeros( self.num_queries, 3, device=queries.device)
                            interaction_mask[:, map_ids] = interaction_mask_matched[batch_id]

                            interaction_mask_list.append(interaction_mask)
                            if self.predict_interaction_centers:
                                interaction_center[map_ids, :] = interaction_centers_matched[batch_id]
                                interaction_centers_list.append(interaction_center)
                        interaction_dict = {
                            "interaction_mask": interaction_mask_list,
                            "interaction_centers": interaction_centers_list,
                            'interaction_mask_vector': interaction_dict_matched["interaction_mask_vector"]
                        }
                    else:
                        interaction_dict = self.hierarchy_interaction_forward(queries_for_interaction, outputs_mask,
                            interaction_features)
                    predictions_interaction_dicts.append(interaction_dict)
                curr_sample_size = max(
                    [pcd.shape[0] for pcd in decomposed_aux]
                )

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError(
                        "only a single point gives nans in cross-attention"
                    )

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(
                        curr_sample_size, self.sample_sizes[hlevel]
                    )

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(
                            pcd_size, device=queries.device
                        )

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(
                            decomposed_aux[k].shape[0], device=queries.device
                        )[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack(
                    [
                        decomposed_aux[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn = torch.stack(
                    [
                        decomposed_attn[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_pos_enc = torch.stack(
                    [
                        pos_encodings_pcd[hlevel][0][k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == rand_idx[0].shape[0]
                ] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                # print(" Decoder: {}, Level: {} batched_aux.shape: {}".format(decoder_counter, i, batched_aux.shape))
                src_pcd = self.lin_squeeze[decoder_counter][i](
                    batched_aux.permute((1, 0, 2))
                )
                
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))
                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)
                
                # refine interaction queries with the output of the current decoder
                if self.predict_hierarchy_interaction and self.use_interaction_queries:
                    # interaction pcd fetures
                    decomposed_aux_interaction = aux_interaction[hlevel].decomposed_features
                    batched_interaction_aux = torch.stack(
                    [
                        decomposed_aux_interaction[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                    )
                    src_interaction_pcd = self.interaction_lin_squeeze[decoder_counter][i](
                        batched_interaction_aux.permute((1, 0, 2))
                    )
                    
                    # interaction attention masks 
                    if self.mov_inter_couple:
                        batched_interaction_attn = batched_attn
                    else:
                        interaction_out_mask = interaction_dict["interaction_mask"]
                        interaction_out_mask = torch.cat(interaction_out_mask)
                        interaction_out_mask = me.SparseTensor(
                            features=interaction_out_mask,
                            coordinate_manager=interaction_features.coordinate_manager,
                            coordinate_map_key=interaction_features.coordinate_map_key,
                        )
                        interaction_attn_mask = interaction_out_mask
                        num_pooling_steps = len(aux) - hlevel - 1
                        for _ in range(num_pooling_steps):
                            interaction_attn_mask = self.pooling(interaction_attn_mask)
                        interaction_attn_mask = me.SparseTensor(
                            features=(interaction_attn_mask.F.detach().sigmoid() < 0.5),
                            coordinate_manager=interaction_attn_mask.coordinate_manager,
                            coordinate_map_key=interaction_attn_mask.coordinate_map_key,
                        )
                        decomposed_interaction_attn_mask = interaction_attn_mask.decomposed_features
                        batched_interaction_attn = torch.stack(
                            [
                                decomposed_interaction_attn_mask[k][rand_idx[k], :]
                                for k in range(len(rand_idx))
                            ]
                        )
                        batched_interaction_attn.permute((0, 2, 1))[
                            batched_interaction_attn.sum(1) == rand_idx[0].shape[0]
                        ] = False
                        batched_interaction_attn = torch.logical_or(batched_interaction_attn, m[..., None])
    
                    # interaction cross attention
                    output_interaction = self.interaction_cross_attention[decoder_counter][i](
                        queries_for_interaction.permute((1, 0, 2)),
                        src_interaction_pcd,
                        memory_mask=batched_interaction_attn.repeat_interleave(
                            self.num_heads, dim=0
                        ).permute((0, 2, 1)),
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=batched_pos_enc.permute((1, 0, 2)),
                        query_pos=interaction_query_pos,
                    )
                    # fuse mov queries with interaction queries
                    output_interaction = output_interaction + self.mov_to_interaction_queries(queries).permute((1, 0, 2))
                    output_interaction = self.interaction_self_attention[decoder_counter][i](
                        output_interaction,
                        tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=interaction_query_pos,
                    )
                    # interaction ffn
                    output_interaction = self.interaction_ffn_attention[decoder_counter][i](
                        output_interaction
                    ).permute((1, 0, 2))
                    # update interaction queries
                    queries_for_interaction = queries_for_interaction + self.interaction_queries_res( output_interaction)
                    pass
                    
                # if not use_gt_movable_mask:
                #     predictions_class.append(output_class)
                #     predictions_mask.append(outputs_mask)
                # else:
                #     predictions_class.append(output_classes_matched)
                #     predictions_mask.append(outputs_masks_matched)
                #     # use indices to select matched origin and axis
                #     if len(predictions_origins) > 0 and len(predictions_axises) > 0:
                #         prediction_origins_matched = []
                #         prediction_axises_matched = []
                #         prediction_origins = predictions_origins[-1]
                #         prediction_axises = predictions_axises[-1]
                #         for batch_id, (map_ids, target_ids) in enumerate(indices):
                #             prediction_origins_matched.append(prediction_origins[batch_id][map_ids])
                #             prediction_axises_matched.append(prediction_axises[batch_id][map_ids])
                #     predictions_origins[-1] = prediction_origins_matched
                #     predictions_axises[-1] = prediction_axises_matched
        if self.train_on_segments:
            output_class, outputs_mask = self.mask_module(
                queries,
                mask_features,
                mask_segments,
                0,
                ret_attn_mask=False,
                point2segment=point2segment,
                coords=coords,
            )
        else:
            output_class, outputs_mask = self.mask_module(
                queries,
                mask_features,
                None,
                0,
                ret_attn_mask=False,
                point2segment=None,
                coords=coords,
            )
        if self.pred_mov_centers:
            pred_mov_centers_out_batch = []
            for bid in range(batch_size):
                coordinates = mask_features.decomposed_coordinates[bid] * self.voxel_size # (point_num, 3)
                pred_mov_centers = self.mov_center_head(mask_features.decomposed_features[bid]) + coordinates # (num_points, 3) 
                pred_mov_centers_expand = pred_mov_centers.unsqueeze(1).expand(-1, self.num_queries, -1) # (num_points, num_queries, 3)
                
                mov_mask_expanded = (outputs_mask[bid].sigmoid()>0.5).unsqueeze(-1).expand(-1, -1, 3) # (num_points, num_queries, 3)
                pred_mov_centers_out = mov_mask_expanded * pred_mov_centers_expand
                pred_mov_centers_out = pred_mov_centers_out.sum(dim=0) / (mov_mask_expanded.sum(dim=0) + 1e-6) # (num_query, 3)
                pred_mov_centers_out_batch.append(pred_mov_centers_out)
            pred_mov_centers_list.append(pred_mov_centers_out_batch)

        if self.predict_articulation:
            outputs_origins, outputs_axises = \
                    self.articulation_forward(queries, outputs_mask, 
                                    arti_features = arti_features,
                                    output_class = output_class)
            predictions_origins.append(outputs_origins)
            predictions_axises.append(outputs_axises)
        if self.predict_hierarchy_interaction:
            if not self.use_interaction_queries:
                queries_for_interaction = queries
                # queries_for_interaction = interaction_queries.unsqueeze(0).repeat(batch_size, 1, 1)
            if use_gt_movable_mask:
                # use ground truth movable mask for interaction prediction
                outputs_level = {
                    "pred_logits": output_class,
                    "pred_masks": outputs_mask,
                }
                indices = matcher(outputs_level, target, "masks")
                # for batch_id, (map_ids, target_ids) in enumerate(indices):
                #     pred_mov_masks_matched = 
                queries_matched = []
                gt_masks_matched = []
                output_classes_matched = []
                output_maskes_gt= []
                for batch_id, (map_ids, target_ids) in enumerate(indices):
                    queries_matched.append(queries_for_interaction[batch_id][map_ids])
                    gt_masks_matched.append(target[batch_id]["masks"][target_ids].T)
                    output_classes_matched.append(output_class[batch_id][map_ids])
                    if use_gt_movable_mask_as_pred:
                        output_maskes_gt_bid = torch.zeros_like(outputs_mask[batch_id]) # (num_points, num_queries)
                        output_maskes_gt_bid[:, map_ids] = target[batch_id]["masks"][target_ids].T.float()
                        output_maskes_gt.append(output_maskes_gt_bid)
                        # num_queries = self.num_queries
                        # gt_masks = target[batch_id]["masks"].T
                        # output_maskes_gt_bid = torch.ones(gt_masks.shape[0], gt_masks.shape[1], device=queries.device).float()
                        # output_maskes_gt_bid[gt_masks] = 1
                        # output_maskes_gt_bid[~gt_masks] = -1
                        # pad = torch.ones(gt_masks.shape[0], num_queries - gt_masks.shape[1], device=queries.device).float() * -1.0
                        # output_maskes_gt_bid = torch.cat([output_maskes_gt_bid, pad], dim=1)
                        # output_maskes_gt.append(output_maskes_gt_bid)   

                if use_gt_movable_mask_as_pred:
                    outputs_mask = output_maskes_gt
                interaction_dict_matched = self.hierarchy_interaction_forward(queries_matched, gt_masks_matched, interaction_features)
                # fill interaction_dict with full num_queries
                interaction_mask_matched = interaction_dict_matched["interaction_mask"]
                interaction_centers_matched = interaction_dict_matched["interaction_centers"]
                interaction_mask_list = []
                interaction_centers_list = []
                for batch_id, (map_ids, target_ids) in enumerate(indices):
                    num_points = interaction_mask_matched[batch_id].shape[0]
                    interaction_mask = torch.zeros( num_points, self.num_queries, device=queries.device)
                    interaction_mask[:, map_ids] = interaction_mask_matched[batch_id]
                    interaction_mask_list.append(interaction_mask)
                    if self.predict_interaction_centers:
                        interaction_center = torch.zeros( self.num_queries, 3, device=queries.device)
                        interaction_center[map_ids, :] = interaction_centers_matched[batch_id]
                        interaction_centers_list.append(interaction_center)
                interaction_dict = {
                        "interaction_mask": interaction_mask_list,
                        "interaction_centers": interaction_centers_list,
                        'interaction_mask_vector': interaction_dict_matched["interaction_mask_vector"]
                    }
            else:
                interaction_dict = self.hierarchy_interaction_forward(queries_for_interaction, outputs_mask,
                    interaction_features)
            predictions_interaction_dicts.append(interaction_dict)
            predictions_class.append(output_class)
            predictions_mask.append(outputs_mask)
            # if not use_gt_movable_mask:
            #     predictions_class.append(output_class)
            #     predictions_mask.append(outputs_mask)
            # else:
            #     predictions_class.append(output_classes_matched)
            #     predictions_mask.append(outputs_masks_matched)
            #     # use indices to select matched origin and axis
            #     if len(predictions_origins) > 0 and len(predictions_axises) > 0:
            #         prediction_origins_matched = []
            #         prediction_axises_matched = []
            #         prediction_origins = predictions_origins[-1]
            #         prediction_axises = predictions_axises[-1]
            #         for batch_id, (map_ids, target_ids) in enumerate(indices):
            #             prediction_origins_matched.append(prediction_origins[batch_id][map_ids])
            #             prediction_axises_matched.append(prediction_axises[batch_id][map_ids])
            #     predictions_origins[-1] = prediction_origins_matched
            #     predictions_axises[-1] = prediction_axises_matched
            
        predictions_interaction_dicts = predictions_interaction_dicts if self.predict_hierarchy_interaction else None
        predictions_origins = predictions_origins if self.predict_articulation else None
        predictions_axises = predictions_axises if self.predict_articulation else None

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask, 
                                              predictions_origins, predictions_axises, 
                                                predictions_interaction_dicts,
                                                pred_mov_centers_list), 
            "sampled_coords": sampled_coords.detach().cpu().numpy()
            if sampled_coords is not None
            else None,
            "backbone_features": pcd_features,
            'pred_origins': predictions_origins[-1] if self.predict_articulation else None,
            'pred_axises': predictions_axises[-1] if self.predict_articulation else None,
            'pred_interaction_dict': predictions_interaction_dicts[-1] if self.predict_hierarchy_interaction else None,
            "pred_mov_centers": pred_mov_centers_list[-1] if self.pred_mov_centers else None
        }

    def mask_module(
        self,
        query_feat,
        mask_features,
        mask_segments,
        num_pooling_steps,
        ret_attn_mask=True,
        point2segment=None,
        coords=None,
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []

        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(
                    mask_features.decomposed_features[i] @ mask_embed[i].T
                )

        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(
            features=output_masks,
            coordinate_manager=mask_features.coordinate_manager,
            coordinate_map_key=mask_features.coordinate_map_key,
        )
        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())

            attn_mask = me.SparseTensor(
                features=(attn_mask.F.detach().sigmoid() < 0.5),
                coordinate_manager=attn_mask.coordinate_manager,
                coordinate_map_key=attn_mask.coordinate_map_key,
            )

            if point2segment is not None:
                return outputs_class, output_segments, attn_mask
            else:
                return (
                    outputs_class,
                    outputs_mask.decomposed_features,
                    attn_mask,
                )

        if point2segment is not None:
            return outputs_class, output_segments
        else:
            return outputs_class, outputs_mask.decomposed_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, 
                      outputs_origins=None, outputs_axises=None,
                      predictions_interaction_dicts=None,
                      pred_mov_centers = None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.pred_mov_centers:
            if self.predict_hierarchy_interaction:
                if self.predict_articulation:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_origins": c, "pred_axises": d, "pred_interaction_dict": e, "pred_mov_centers": f}
                        for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_origins[:-1], outputs_axises[:-1], predictions_interaction_dicts[:-1], pred_mov_centers[:-1])
                    ]
                else:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_interaction_dict": e, "pred_mov_centers": f}
                        for a, b, e, f in zip(outputs_class[:-1], outputs_seg_masks[:-1], predictions_interaction_dicts[:-1], pred_mov_centers[:-1])
                    ]
            else:
                if self.predict_articulation:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_origins": c, "pred_axises": d, "pred_mov_centers": f}
                        for a, b, c, d, f in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_origins[:-1], outputs_axises[:-1], pred_mov_centers[:-1])
                    ]
                else:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_mov_centers": f}
                        for a, b, f in zip(outputs_class[:-1], outputs_seg_masks[:-1], pred_mov_centers[:-1])
                ]
        else:
            if self.predict_hierarchy_interaction:
                if self.predict_articulation:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_origins": c, "pred_axises": d, "pred_interaction_dict": e}
                        for a, b, c, d, e in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_origins[:-1], outputs_axises[:-1], predictions_interaction_dicts[:-1])
                    ]
                else:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_interaction_dict": e}
                        for a, b, e in zip(outputs_class[:-1], outputs_seg_masks[:-1], predictions_interaction_dicts[:-1])
                    ]
            else:
                if self.predict_articulation:
                    return [
                        {"pred_logits": a, "pred_masks": b, "pred_origins": c, "pred_axises": d}
                        for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_origins[:-1], outputs_axises[:-1])
                    ]
                else:
                    return [
                        {"pred_logits": a, "pred_masks": b}
                        for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )

class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )

class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
