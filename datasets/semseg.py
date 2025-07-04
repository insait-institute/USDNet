import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange
import h5py

import numpy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml
import pickle

# from yaml import CLoader as Loader
from torch.utils.data import Dataset
from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
)
from datasets.scannetpp.scannetpp_constants import (SCANNETPP_COLOR_MAP)

MULTISCAN_COLOR_MAP = {0: (0.0, 0.0, 0.0),
 1: (191, 246, 112),
 2: (110, 239, 148),
 255: (0.0, 0.0, 0.0),}
SCENEFUN3D_COLOR_MAP = MULTISCAN_COLOR_MAP
ARTICULATE3D_COLOR_MAP = MULTISCAN_COLOR_MAP

logger = logging.getLogger(__name__)

from scipy.spatial import cKDTree

class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=False,
        color_drop=0.0,
        epoch = 0,
        use_coarse_to_fine: bool = False,
        c2f_alpha = 200,
        c2f_decay = 0.5,
        c2f_rad = 0.1,
        use_hierarchy = False,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "unknown task"

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "stpls3d":
            self.color_map = {
                0: [0, 255, 0],  # Ground
                1: [0, 0, 255],  # Build
                2: [0, 255, 255],  # LowVeg
                3: [255, 255, 0],  # MediumVeg
                4: [255, 0, 255],  # HiVeg
                5: [100, 100, 255],  # Vehicle
                6: [200, 200, 100],  # Truck
                7: [170, 120, 200],  # Aircraft
                8: [255, 0, 0],  # MilitaryVec
                9: [200, 100, 100],  # Bike
                10: [10, 200, 100],  # Motorcycle
                11: [200, 200, 200],  # LightPole
                12: [50, 50, 50],  # StreetSign
                13: [60, 130, 60],  # Clutter
                14: [130, 30, 60],
            }  # Fence
        elif self.dataset_name == "scannet200":
            self.color_map = SCANNET_COLOR_MAP_200
        elif self.dataset_name == "s3dis":
            self.color_map = {
                0: [0, 255, 0],  # ceiling
                1: [0, 0, 255],  # floor
                2: [0, 255, 255],  # wall
                3: [255, 255, 0],  # beam
                4: [255, 0, 255],  # column
                5: [100, 100, 255],  # window
                6: [200, 200, 100],  # door
                7: [170, 120, 200],  # table
                8: [255, 0, 0],  # chair
                9: [200, 100, 100],  # sofa
                10: [10, 200, 100],  # bookcase
                11: [200, 200, 200],  # board
                12: [50, 50, 50],  # clutter
            }
        elif self.dataset_name == "scannetpp":
            self.color_map = SCANNETPP_COLOR_MAP
        elif self.dataset_name == "multiscan":
            self.color_map = MULTISCAN_COLOR_MAP
        elif self.dataset_name == "scenefun3d":
            self.color_map = SCENEFUN3D_COLOR_MAP
        elif self.dataset_name == "articulate3d":
            self.color_map = ARTICULATE3D_COLOR_MAP
        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(
            self.crop_min_size,
            crop_length=self.crop_length,
            version1=self.version1,
        )

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points
        self.epoch = epoch
        
        self.use_coarse_to_fine = use_coarse_to_fine
        if isinstance(self.use_coarse_to_fine, str):
            self.use_coarse_to_fine = self.use_coarse_to_fine.lower() == "true"
        print("self.use_coarse_to_fine: ", self.use_coarse_to_fine)
        print("type(self.use_coarse_to_fine): ", type(self.use_coarse_to_fine))
        self.use_hierarchy = use_hierarchy
        if isinstance(self.use_hierarchy, str):
            self.use_hierarchy = self.use_hierarchy.lower() == "true"
        print("self.use_hierarchy: ", self.use_hierarchy)
        print("type(self.use_hierarchy): ", type(self.use_hierarchy))
        self.c2f_alpha = c2f_alpha
        self.c2f_decay = c2f_decay
        self.c2f_rad = c2f_rad
        self.expanded_masks = {}

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if self.dataset_name != "s3dis":
                if not (database_path / f"{mode}_database.yaml").exists():
                    print(
                        f"generate {database_path}/{mode}_database.yaml first"
                    )
                    exit()
                self._data.extend(
                    self._load_yaml(database_path / f"{mode}_database.yaml")
                )
            else:
                mode_s3dis = f"Area_{self.area}"
                if self.mode == "train":
                    mode_s3dis = "train_" + mode_s3dis
                if not (
                    database_path / f"{mode_s3dis}_database.yaml"
                ).exists():
                    print(
                        f"generate {database_path}/{mode_s3dis}_database.yaml first"
                    )
                    exit()
                self._data.extend(
                    self._load_yaml(
                        database_path / f"{mode_s3dis}_database.yaml"
                    )
                )
        if data_percent < 1.0:
            self._data = sample(
                self._data, int(len(self._data) * data_percent)
            )
        labels = self._load_yaml(Path(label_db_filepath))

        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)

        if instance_oversampling > 0:
            self.instance_data = self._load_yaml(
                Path(label_db_filepath).parent / "instance_database.yaml"
            )

        # normalize color channels
        if self.dataset_name == "s3dis":
            color_mean_std = color_mean_std.replace(
                "color_mean_std.yaml", f"Area_{self.area}_color_mean_std.yaml"
            )

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error(
                "pass mean and std as tuple of tuples, or as an .yaml file"
            )

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data
        # new_data = []
        if self.cache_data:
            new_data = []
            for i in range(len(self._data)):
                self._data[i]["data"] = np.load(
                    self.data[i]["filepath"].replace("../../", "")
                )
                if self.on_crops:
                    if self.eval_inner_core == -1:
                        for block_id, block in enumerate(
                            self.splitPointCloud(self._data[i]["data"])
                        ):
                            if len(block) > 10000:
                                new_data.append(
                                    {
                                        "instance_gt_filepath": self._data[i][
                                            "instance_gt_filepath"
                                        ][block_id]
                                        if len(
                                            self._data[i][
                                                "instance_gt_filepath"
                                            ]
                                        )
                                        > 0
                                        else list(),
                                        "scene": f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
                                        "raw_filepath": f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
                                        "data": block,
                                    }
                                )
                            else:
                                assert False
                    else:
                        conds_inner, blocks_outer = self.splitPointCloud(
                            self._data[i]["data"],
                            size=self.crop_length,
                            inner_core=self.eval_inner_core,
                        )

                        for block_id in range(len(conds_inner)):
                            cond_inner = conds_inner[block_id]
                            block_outer = blocks_outer[block_id]

                            if cond_inner.sum() > 10000:
                                new_data.append(
                                    {
                                        "instance_gt_filepath": self._data[i][
                                            "instance_gt_filepath"
                                        ][block_id]
                                        if len(
                                            self._data[i][
                                                "instance_gt_filepath"
                                            ]
                                        )
                                        > 0
                                        else list(),
                                        "scene": f"{self._data[i]['scene'].replace('.txt', '')}_{block_id}.txt",
                                        "raw_filepath": f"{self.data[i]['filepath'].replace('.npy', '')}_{block_id}",
                                        "data": block_outer,
                                        "cond_inner": cond_inner,
                                    }
                                )
                            else:
                                assert False

            if self.on_crops:
                self._data = new_data
                # new_data.append(np.load(self.data[i]["filepath"].replace("../../", "")))
            # self._data = new_data

        # load expanded masks
        if self.use_coarse_to_fine:
            self.expanded_masks = {}
            for item in self._data:
                self.expanded_masks[item["scene"]] = {}
                expand_mask_dict_file = item["expand_dict_file"]
                # load pickle 
                with open(expand_mask_dict_file, "rb") as f:
                    self.expanded_masks[item["scene"]] = pickle.load(f)
            

    def set_epoch(self, epoch):
        self.epoch = epoch

    def splitPointCloud(self, cloud, size=50.0, stride=50, inner_core=-1):
        if inner_core == -1:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - size) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
            cells = [
                (x * stride, y * stride)
                for x in range(width)
                for y in range(depth)
            ]
            blocks = []
            for (x, y) in cells:
                xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
                ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
                cond = xcond & ycond
                block = cloud[cond, :]
                blocks.append(block)
            return blocks
        else:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - inner_core) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - inner_core) / stride)) + 1
            cells = [
                (x * stride, y * stride)
                for x in range(width)
                for y in range(depth)
            ]
            blocks_outer = []
            conds_inner = []
            for (x, y) in cells:
                xcond_outer = (
                    cloud[:, 0] <= x + inner_core / 2.0 + size / 2
                ) & (cloud[:, 0] >= x + inner_core / 2.0 - size / 2)
                ycond_outer = (
                    cloud[:, 1] <= y + inner_core / 2.0 + size / 2
                ) & (cloud[:, 1] >= y + inner_core / 2.0 - size / 2)

                cond_outer = xcond_outer & ycond_outer
                block_outer = cloud[cond_outer, :]

                xcond_inner = (block_outer[:, 0] <= x + inner_core) & (
                    block_outer[:, 0] >= x
                )
                ycond_inner = (block_outer[:, 1] <= y + inner_core) & (
                    block_outer[:, 1] >= y
                )

                cond_inner = xcond_inner & ycond_inner

                conds_inner.append(cond_inner)
                blocks_outer.append(block_outer)
            return conds_inner, blocks_outer

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        if self.is_tta:
            return 5 * len(self.data)
        else:
            return self.reps_per_epoch * len(self.data)
        
    def expand_instances_and_semantics(self, scene_idx, points, instance_ids, sem_ids, radius, ignore_index = 0):
        if scene_idx not in self.expanded_masks:
            self.expanded_masks[scene_idx] = {}
            expand_idx_records = []
            expand_inst_records = []
            expand_sem_records = []
            expand_distances = []
            
            # Initialize expanded instance and semantic IDs with original assignments
            expanded_instance_ids = instance_ids.copy()
            expanded_sem_ids = sem_ids.copy()
            # Create a k-d tree for neighborhood queries, only expand points to background
            bg_mask = instance_ids == ignore_index
            points_bg = points[bg_mask]
            tree = cKDTree(points_bg)
            # Iterate over each unique instance ID
            for instance_id in np.unique(instance_ids):
                if instance_id == ignore_index:
                    continue
                # Mask for points in the current instance
                instance_mask = instance_ids == instance_id
                instance_points = points[instance_mask]
                instance_sem_id = sem_ids[instance_mask][0]  # Assume all points in an instance have the same sem_id
                # Expand points for this instance
                for idx, point in zip(np.where(instance_mask)[0], instance_points):
                    neighbors_idx = tree.query_ball_point(point, radius)
                    for neighbor_idx in neighbors_idx:
                        distance = np.min(np.linalg.norm(points[instance_mask] - points_bg[neighbor_idx], axis=1))
                        cur_id = expanded_instance_ids[bg_mask][neighbor_idx]
                        # If the neighbor is closer to the current instance than its current assigned instance
                        not_assigned = cur_id == ignore_index
                        if not_assigned:
                            # Update instance and semantic IDs
                            indices_to_update = np.where(bg_mask)[0][neighbor_idx]
                            expanded_instance_ids[indices_to_update] = instance_id
                            expanded_sem_ids[indices_to_update] = instance_sem_id
                            ## record 
                            expand_idx_records.append(indices_to_update)
                            expand_inst_records.append(instance_id)
                            expand_sem_records.append(instance_sem_id)
                            expand_distances.append(distance)
                        else:
                            # If the neighbor is closer to the current instance than its current assigned instance
                            ## calculate the distance to the current instance
                            # Calculate distances
                            if cur_id != instance_id:
                                cur_instance_mask = instance_ids == cur_id
                                cur_instance_points = points[cur_instance_mask]
                                distance_cur = np.min(np.linalg.norm(cur_instance_points - points_bg[neighbor_idx], axis=1))
                                if distance < distance_cur:
                                    # Update instance and semantic IDs
                                    indices_to_update = np.where(bg_mask)[0][neighbor_idx]
                                    expanded_instance_ids[indices_to_update] = instance_id
                                    expanded_sem_ids[indices_to_update] = instance_sem_id
                                    
                                    ## update records
                                    record_idx = expand_idx_records.index(indices_to_update)
                                    expand_inst_records[record_idx] = instance_id
                                    expand_sem_records[record_idx] = instance_sem_id
                                    expand_distances[record_idx] = distance
            expand_idx_records = np.array(expand_idx_records)
            expand_inst_records = np.array(expand_inst_records)
            expand_sem_records = np.array(expand_sem_records)
            expand_distances = np.array(expand_distances)
            self.expanded_masks[scene_idx] = {
                "expand_idx_records": expand_idx_records,
                "expand_inst_records": expand_inst_records,
                "expand_sem_records": expand_sem_records,
                "expand_distances": expand_distances
            }
        else:
            valid_mask = self.expanded_masks[scene_idx]["expand_distances"] <= radius
            expand_idx_records = self.expanded_masks[scene_idx]["expand_idx_records"][valid_mask]
            expand_inst_records = self.expanded_masks[scene_idx]["expand_inst_records"][valid_mask]
            expand_sem_records = self.expanded_masks[scene_idx]["expand_sem_records"][valid_mask]
            # expand instance and semantic IDs
            expanded_instance_ids = np.copy(instance_ids)
            expanded_sem_ids = np.copy(sem_ids)
            expanded_instance_ids[expand_idx_records] = expand_inst_records
            expanded_sem_ids[expand_idx_records] = expand_sem_records
                                    
        return expanded_instance_ids, expanded_sem_ids

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        if self.cache_data:
            points = self.data[idx]["data"]
        else:
            assert not self.on_crops, "you need caching if on crops"
            points = np.load(self.data[idx]["filepath"].replace("../../", ""))

        if "train" in self.mode and self.dataset_name in ["s3dis", "stpls3d"]:
            inds = self.random_cuboid(points)
            points = points[inds]

        if self.dataset_name == 'scannetpp' \
            or self.dataset_name == 'multiscan' \
            or self.dataset_name == 'scenefun3d':
            coordinates, color, normals, labels, segments = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6:9],
                points[:, 9:11],
                points[:, 11],
            )
        elif self.dataset_name == 'articulate3d':
            coordinates, color, normals, labels, segments, inter_ids = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6:9],
                points[:, 9:11],
                points[:, 11],
                points[:, 12],)
            # print(" inter_ids: ", np.unique(inter_ids))
        else:
            coordinates, color, normals, segments, labels = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6:9],
                points[:, 9],
                points[:, 10:12],
            )
        if self.use_coarse_to_fine:
            radius_epoch = self.c2f_rad * \
                self.c2f_decay ** np.floor(self.epoch / self.c2f_alpha)
            # print(" Epoch {}, radius_epoch: {}".format(self.epoch, radius_epoch))
            # expanded_instance_ids, expanded_sem_ids = self.expand_instances_and_semantics(
            #     idx, coordinates, labels[:, 1], labels[:, 0], radius_epoch) 
            scene_id = self.data[idx]["scene"]
            valid_mask = self.expanded_masks[scene_id]["expand_distances"] <= radius_epoch
            expand_idx_records = self.expanded_masks[scene_id]["expand_idx_records"][valid_mask]
            expand_inst_records = self.expanded_masks[scene_id]["expand_inst_records"][valid_mask]
            expand_sem_records = self.expanded_masks[scene_id]["expand_sem_records"][valid_mask]
            expanded_instance_ids = np.copy(labels[:, 1])
            expanded_sem_ids = np.copy(labels[:, 0])
            # print("expand_idx_records: ", expand_idx_records)
            # print(" dtype of expand_idx_records: ", expand_idx_records.dtype)
            if expand_idx_records.shape[0] > 0:
                expanded_instance_ids[expand_idx_records] = expand_inst_records
                expanded_sem_ids[expand_idx_records] = expand_sem_records
                labels = np.hstack((expanded_sem_ids[..., None],            expanded_instance_ids[..., None]))
                if self.dataset_name == 'articulate3d':
                    inter_ids[expand_idx_records] = expand_inst_records
            
        # print("step 1, labels.shape", labels.shape)
        # print("unique semseg labels 1", np.unique(labels[:, 0]))
        # print("original insts", np.unique(labels[:, 1]))
        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        shift = np.zeros(3)
        flip_axis = {}
        if "train" in self.mode or self.is_tta:
            if self.cropping:
                # new_idx = self.random_cuboid(
                #     coordinates,
                #     labels[:, 1],
                #     self._remap_from_zero(labels[:, 0].copy()),
                # )
                new_idx = self.random_cuboid(coordinates)
                coordinates = coordinates[new_idx]
                color = color[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]
                points = points[new_idx]
                if self.dataset_name == 'articulate3d':
                    inter_ids = inter_ids[new_idx]
                # inter_ids = inter_ids[new_idx]

            # coordinates -= coordinates.mean(0)
            shift = -coordinates.mean(0)
            try:
                shift += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0))
                    / 2
                )
                coordinates += shift
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            if self.instance_oversampling > 0.0:
                (
                    coordinates,
                    color,
                    normals,
                    labels,
                ) = self.augment_individual_instance(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.instance_oversampling,
                )

            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            # for i in (0, 1):
            #     if random() < 0.5:
            #         coord_max = np.max(points[:, i])
            #         coordinates[:, i] = coord_max - coordinates[:, i]
            #         flip_axis[i] = coord_max

            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            aug = self.volume_augmentations(
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            # print("step 2, labels.shape", labels.shape)
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

            if self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        labels[~indexes],
                    )

            # if self.noise_rate > 0:
            #     coordinates, color, normals, labels = random_points(
            #         coordinates,
            #         color,
            #         normals,
            #         labels,
            #         self.noise_rate,
            #         self.ignore_label,
            #     )

            if (self.resample_points > 0) or (self.noise_rate > 0):
                coordinates, color, normals, labels = random_around_points(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.resample_points,
                    self.noise_rate,
                    self.ignore_label,
                )

            if self.add_unlabeled_pc:
                if random() < 0.8:
                    new_points = np.load(
                        self.other_database[
                            np.random.randint(0, len(self.other_database) - 1)
                        ]["filepath"]
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        new_points[:, :3],
                        new_points[:, 3:6],
                        new_points[:, 6:9],
                        new_points[:, 9:],
                    )
                    unlabeled_coords -= unlabeled_coords.mean(0)
                    unlabeled_coords += (
                        np.random.uniform(
                            unlabeled_coords.min(0), unlabeled_coords.max(0)
                        )
                        / 2
                    )

                    aug = self.volume_augmentations(
                        points=unlabeled_coords,
                        normals=unlabeled_normals,
                        features=unlabeled_color,
                        labels=unlabeled_labels,
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        aug["points"],
                        aug["features"],
                        aug["normals"],
                        aug["labels"],
                    )
                    pseudo_image = unlabeled_color.astype(np.uint8)[
                        np.newaxis, :, :
                    ]
                    unlabeled_color = np.squeeze(
                        self.image_augmentations(image=pseudo_image)["image"]
                    )

                    coordinates = np.concatenate(
                        (coordinates, unlabeled_coords)
                    )
                    color = np.concatenate((color, unlabeled_color))
                    normals = np.concatenate((normals, unlabeled_normals))
                    labels = np.concatenate(
                        (
                            labels,
                            np.full_like(unlabeled_labels, self.ignore_label),
                        )
                    )

            if random() < self.color_drop:
                color[:] = 255

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        # print("step 3, labels.shape", labels.shape)
        # prepare labels and map from 0 to 20(40)
        # print("unique semseg labels 2", np.unique(labels[:, 0]))
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]
        # print("step 4, labels.shape", labels.shape)
        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        # if self.task != "semantic_segmentation":
        if self.data[idx]["raw_filepath"].split("/")[-2] in [
            "scene0636_00",
            "scene0154_00",
        ]:
            return self.__getitem__(0)
        # print("unique semseg labels", np.unique(labels[:, 0]))
        # print("precessed insts", np.unique(labels[:, 1]))
        if self.dataset_name == "s3dis":
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["area"] + "_" + self.data[idx]["scene"],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
            )
        if self.dataset_name == "stpls3d":
            if labels.shape[1] != 1:  # only segments --> test set!
                if np.unique(labels[:, -2]).shape[0] < 2:
                    print("NO INSTANCES")
                    return self.__getitem__(0)
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["scene"],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
            )
        elif self.dataset_name == "scannetpp":
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["raw_filepath"].split("/")[-1],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
            )
        elif self.dataset_name == "multiscan":
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["raw_filepath"].split("/")[-1],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
                (shift, flip_axis)
            )
        elif self.dataset_name == "scenefun3d":
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["raw_filepath"].split("/")[-1],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
                (shift, flip_axis)
            )
        elif self.dataset_name == "articulate3d":
            if self.use_hierarchy:
                return (
                    coordinates,
                    features,
                    labels,
                    self.data[idx]["raw_filepath"].split("/")[-1],
                    raw_color,
                    raw_normals,
                    raw_coordinates,
                    idx,
                    (shift, flip_axis),
                    inter_ids
                )
            else:
                return (
                    coordinates,
                    features,
                    labels,
                    self.data[idx]["raw_filepath"].split("/")[-1],
                    raw_color,
                    raw_normals,
                    raw_coordinates,
                    idx,
                    (shift, flip_axis)                )
        else:
            return (
                coordinates,
                features,
                labels,
                self.data[idx]["raw_filepath"].split("/")[-2],
                raw_color,
                raw_normals,
                raw_coordinates,
                idx,
            )

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped

    def augment_individual_instance(
        self, coordinates, color, normals, labels, oversampling=1.0
    ):
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[
                        labels[:, 1] == choice(np.unique(labels[:, 1]))
                    ]
                )
            else:
                center = np.array(
                    [uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)]
                )
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = (
                instance[:, :3] - instance[:, :3].mean(axis=0) + center
            )
            max_instance = max_instance + 1
            instance[:, -1] = max_instance
            aug = V.Compose(
                [
                    V.Scale3d(),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 24, axis=(1, 0, 0)
                    ),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 24, axis=(0, 1, 0)
                    ),
                    V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
                ]
            )(
                points=instance[:, :3],
                features=instance[:, 3:6],
                normals=instance[:, 6:9],
                labels=instance[:, 9:],
            )
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"]))

        return coordinates, color, normals, labels


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(
            noise, blurx, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blury, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blurz, mode="constant", cval=0
        )

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])[
            "points"
        ]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])[
            "points"
        ]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates,
    color,
    normals,
    labels,
    rate=0.2,
    noise_rate=0,
    ignore_label=255,
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int(
        (max(max_boundary) - min(min_boundary)) / noise_rate
    )

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(
                    min_boundary[0], max_boundary[0], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[1], max_boundary[1], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[2], max_boundary[2], noisy_coordinates
                ),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full(
        (noisy_coordinates.shape[0], labels.shape[1]), ignore_label
    )

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels

# Recursive function to load nested dictionaries from an HDF5 group
def load_dict_from_h5group(h5group):
    data_dict = {}
    for key, item in h5group.items():
        # Try to convert the key back to an integer if possible
        try:
            int_key = int(key)
        except ValueError:
            int_key = key  # If conversion fails, keep the key as a string

        if isinstance(item, h5py.Group):
            # Recursively load groups as dictionaries
            data_dict[int_key] = load_dict_from_h5group(item)
        else:
            # Load datasets directly
            data = item[()]
            # # Convert NumPy arrays back to lists if needed
            # if isinstance(data, np.ndarray):
            #     data = data.tolist()
            data_dict[int_key] = data
    return data_dict

# Load the dictionary from the HDF5 file
def load_dict_from_h5file(file_path):
    with h5py.File(file_path, 'r') as h5file:
        return load_dict_from_h5group(h5file)

class SemanticSegmentationArticulationDataset(SemanticSegmentationDataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=False,
        color_drop=0.0,
        load_articulation=False,
        epoch=0,
        use_coarse_to_fine: bool = False,
        c2f_alpha = 200,
        c2f_decay = 0.5,
        c2f_rad = 0.1, 
        use_hierarchy=False,
    ):
        super().__init__(dataset_name, data_dir, label_db_filepath,             
                         color_mean_std, 
                         mode, add_colors, add_normals, add_raw_coordinates, add_instance, 
                         num_labels, data_percent, ignore_label, volume_augmentations_path, 
                         image_augmentations_path, instance_oversampling, place_around_existing, 
                         max_cut_region, point_per_cut, flip_in_center, noise_rate, 
                         resample_points, cache_data, add_unlabeled_pc, task, cropping, 
                         cropping_args, is_tta, crop_min_size, crop_length, cropping_v1, 
                         reps_per_epoch, area, on_crops, eval_inner_core, 
                         filter_out_classes, label_offset, add_clip, 
                         is_elastic_distortion, color_drop, epoch, use_coarse_to_fine,
                         c2f_alpha, c2f_decay, c2f_rad, 
                         use_hierarchy)
        self.load_articulation = load_articulation
        
    def __getitem__(self, idx: int):
        if self.load_articulation:
            if self.use_hierarchy:
                coordinates, features, labels, scene, raw_color, raw_normals, \
                raw_coordinates, idx, coord_aug_info, interaction_ids \
                = super().__getitem__(idx)
            else:
                coordinates, features, labels, scene, raw_color, raw_normals, raw_coordinates, idx, coord_aug_info \
                    = super().__getitem__(idx)
            articulation_file = self.data[idx]['articulation_gt_file']
            arti_anno = load_dict_from_h5file(articulation_file)
            if 'train' in self.mode:
                # translate the origin 
                shift, flip_axis = coord_aug_info 
                for inst_id in arti_anno:
                    arti_anno[inst_id]['origin'] += shift
                    # flip 
                    for axis in flip_axis:
                        arti_anno[inst_id]['origin'][axis] = flip_axis[axis] - arti_anno[inst_id]['origin'][axis]
                        arti_anno[inst_id]['axis'] = - arti_anno[inst_id]['axis']
            # get center of each movalbe parts 
            mov_parts_centers = {}
            for inst_id in arti_anno:
                mov_mask = labels[:, 1] == inst_id
                mask_coords = coordinates[mov_mask]
                if mask_coords.shape[0] == 0:
                    mov_center = np.zeros(3)
                    # logger.warning( "movable part {} has no points".format(inst_id))
                else:
                    mov_center = mask_coords.mean(0)
                    mov_parts_centers[inst_id] = mov_center
            if self.use_hierarchy:
                interaction_centers = {}
                for inst_id in arti_anno:
                    # get interaction_mask
                    interaction_mask = interaction_ids == inst_id
                    mask_coords = coordinates[interaction_mask]
                    if mask_coords.shape[0] == 0:
                        inter_center = np.zeros(3)
                    else:
                        inter_center = coordinates[interaction_mask].mean(0)
                    interaction_centers[inst_id] = inter_center
                # print("interaction_centers: ", interaction_centers)
                return coordinates, features, labels, scene, raw_color, raw_normals, raw_coordinates, \
                    idx, arti_anno, interaction_ids, interaction_centers, mov_parts_centers
            else:
                return coordinates, features, labels, scene, raw_color, raw_normals, raw_coordinates, idx, arti_anno, mov_parts_centers
            
        else:
            return super().__getitem__(idx)
            