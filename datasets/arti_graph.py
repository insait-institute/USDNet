import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange
import h5py
import MinkowskiEngine as ME
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

logger = logging.getLogger(__name__)

from scipy.spatial import cKDTree

def to_tensor(data):
    if isinstance(data, dict):
        return {key: to_tensor(val) for key, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_tensor(val) for val in data]
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return data
    
MULTISCAN_COLOR_MAP = {0: (0.0, 0.0, 0.0),
 1: (191, 246, 112),
 2: (110, 239, 148),
 255: (0.0, 0.0, 0.0),}
SCENEFUN3D_COLOR_MAP = MULTISCAN_COLOR_MAP
ARTICULATE3D_COLOR_MAP = MULTISCAN_COLOR_MAP

class ArtiConnectGraphDataset(Dataset):
    def __init__(
        self,
        dataset_name="articulate3d_connect_graph",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/articulate3d_arti_graph",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
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
        task="connectivity_graph_prediction",
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
    ):
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
        
        # loading database files
        self._data = []
        db_file_path = Path(data_dir) / f"{mode}_database.yaml"
        assert db_file_path.exists(), f"Database file {db_file_path} does not exist"
        self._data.extend(self._load_yaml(db_file_path))
        
        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop
        
        if self.dataset_name == "articulate3d_connect_graph":
            self.color_map = ARTICULATE3D_COLOR_MAP
        else:
            assert False, "dataset not known"
            
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if not (database_path / f"{mode}_database.yaml").exists():
                print(
                    f"generate {database_path}/{mode}_database.yaml first"
                )
                exit()
            self._data.extend(
                self._load_yaml(database_path / f"{mode}_database.yaml")
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
        
    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)
        points = np.load(self.data[idx]["filepath"].replace("../../", ""))
        
        if self.dataset_name == 'articulate3d_connect_graph':
            coordinates, colors, normals, mov_gt, inter_gt = (
                    points[:, :3],
                    points[:, 3:6],
                    points[:, 6:9],
                    points[:, 9],
                    points[:, 10],
                )
        else:
            assert False, "dataset {} not known".format(self.dataset_name)
            
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
                new_idx = self.random_cuboid(coordinates)
                coordinates = coordinates[new_idx]
                color = color[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]
                points = points[new_idx]
                
                mov_gt = mov_gt[new_idx]
                inter_gt = inter_gt[new_idx]
            
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
            
            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)
            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
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
                    coordinates, normals, color = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                    )
                    mov_gt = mov_gt[~indexes]
                    inter_gt = inter_gt[~indexes]
        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        mov_gt = mov_gt.astype(np.int32)
        inter_gt = inter_gt.astype(np.int32)
        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
        return (
            coordinates,
            features,
            mov_gt,
            inter_gt,
            self.data[idx]["raw_filepath"].split("/")[-1],
            raw_color,
            raw_normals,
            raw_coordinates,
            idx,
            (shift, flip_axis),
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
    
class VoxelizeGraphCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
        load_articulation=False,
        use_hierarchy=False,
    ):
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold
        self.load_articulation = load_articulation
        self.num_queries = num_queries
        self.use_hierarchy = use_hierarchy

    def __call__(self, batch):
        pass

def voxelize(
    batch,
    ignore_label,
    voxel_size,
    probing,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    num_queries,
):
    (
        coordinates,
        features,
        mov_gts,
        inter_gts,
        original_mov_gts,
        original_inter_gts,
        inverse_maps,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
    ) = ([], [], [], [], [], [], [], [], [])
    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []

    for sample in batch:
        idx.append(sample[8])
        original_coordinates.append(sample[7])
        original_mov_gts.append(sample[2])
        original_inter_gts.append(sample[3])
        full_res_coords.append(sample[0])
        original_colors.append(sample[5])
        original_normals.append(sample[6])
        coords = np.floor(sample[0] / voxel_size)
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample[1],
            }
        )

        # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            **voxelization_dict
        )
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        sample_mov_gt = sample[2][unique_map]
        mov_gts.append(torch.from_numpy(sample_mov_gt).long())
        sample_inter_gt = sample[3][unique_map]
        inter_gts.append(torch.from_numpy(sample_inter_gt).long())
        
    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    input_dict["mov_gt"] = mov_gts
    input_dict["inter_gt"] = inter_gts
    coordinates, features, mov_gts, inter_gts = ME.utils.sparse_collate(**input_dict)

    mv_target, inetr_target = [], []
    mv_target_full, inter_target_full = [], []
    mv_target = get_instance_masks(mov_gts)
    inetr_target = get_instance_masks(inter_gts)
    
    return (mv_target, inetr_target, 
            NoGpu(coordinates, features),[sample[3] for sample in batch])
        
class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        mv_target=None,
        inetr_target=None,
        idx=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        # self.mv_target = mv_target
        # self.inetr_target = inetr_target
        self.idx = idx


def get_instance_masks(
    gts_labels,
    ignore_class_threshold=100,
    label_offset=0,
):
    target = []

    for batch_id in range(len(gts_labels)):
        sem_ids = []
        inst_ids = []
        masks = []

        unique_labels = gts_labels[batch_id].unique()
        selected_unique_labels = []
        for unique_label in unique_labels:
            if unique_label == -1 or unique_label == 0:
                continue
            sem_id = unique_label // 1000
            inst_id = unique_label % 1000
            
            sem_ids.append(sem_id)
            inst_ids.append(inst_id)

            masks.append(gts_labels[batch_id] == unique_label)

        if len(sem_ids) == 0:
            return list()

        sem_ids = torch.stack(sem_ids)
        inst_ids = torch.stack(inst_ids) # shape (num_instances, )
        masks = torch.stack(masks)
    
        sem_ids = torch.clamp(sem_ids - label_offset, min=0)
        batch_anno = {
                    "sem_ids": sem_ids,
                    "inst_ids": inst_ids,
                    "masks": masks,}
        target.append(batch_anno)
    return target

