import re, os, json
from pathlib import Path
from collections import OrderedDict, defaultdict, Counter
import os.path as osp
import numpy as np
import pandas as pd
import h5py
import open3d as o3d
from fire import Fire
from natsort import natsorted
from loguru import logger
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals
import pickle
from copy import deepcopy

# Recursive function to save nested dictionaries to an HDF5 file
def save_dict_to_h5group(h5file, data_dict, h5group):
    for key, value in data_dict.items():
        # Convert non-string keys to strings (like integers)
        if not isinstance(key, str):
            key = str(key)
            
        if isinstance(value, dict):
            # If the value is a dictionary, create a subgroup
            subgroup = h5group.create_group(key)
            save_dict_to_h5group(h5file, value, subgroup)  # Recursively save the nested dictionary
        elif isinstance(value, list):
            # If the value is a list, convert it to a NumPy array
            h5group.create_dataset(key, data=np.array(value))
        elif isinstance(value, str):
            # Handle string data separately
            dt = h5py.string_dtype(encoding='utf-8')
            h5group.create_dataset(key, data=value, dtype=dt)
        else:
            # Save arrays or scalar values directly
            h5group.create_dataset(key, data=value)
# Function to save the dictionary to an HDF5 file
def save_dict_to_h5file(data_dict, file_path):
    with h5py.File(file_path, 'w') as h5file:
        save_dict_to_h5group(h5file, data_dict, h5file)

CLASS_LABELS = ['background', 'rotation','translation']
CLASS_IDS = [0, 1, 2]
VALID_CLASS_IDS = [1, 2]
INSTANCE_LABELS = ['rotation','translation']
LABEL2ID = {CLASS_LABELS[i]: label for i, label in enumerate(CLASS_IDS)}
sem_label_to_id = {
    'background': 0,
    'rot': 1,
    'trans': 2,
}
sem_id_to_label = {v: k for k, v in sem_label_to_id.items()}
SCENEFUN3D_COLOR_MAP = {0: (0.0, 0.0, 0.0),
 1: (191, 246, 112),
 2: (110, 239, 148),
 255: (0.0, 0.0, 0.0),}

splits = {'train': "train_scenes.txt",
          'val': "val_scenes.txt"}

def downsample(points, sem_gt, inst_gt, voxel_size=0.01, ignore_index=0):
    coords = points[:, :3]
    colors = points[:, 3:6]
    normals = points[:, 6:9]
    total_downsample_points = 0
    points_list = []
    colors_list = []
    normals_list = []
    sem_list = []
    inst_list = []
    insts = np.unique(inst_gt)
    for inst in insts:
        inst_mask = inst_gt == inst
        inst_coords = coords[inst_mask]
        inst_colors = colors[inst_mask]
        inst_normals = normals[inst_mask]
        sem_id = sem_gt[inst_mask][0]
        
        inst_pcd = o3d.geometry.PointCloud()
        inst_pcd.points = o3d.utility.Vector3dVector(inst_coords)
        inst_pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        
        if sem_id == ignore_index:
            downinst_pcd = inst_pcd.voxel_down_sample(voxel_size=voxel_size)
        else:
            downinst_pcd = inst_pcd
        
        points_list.append(np.asarray(downinst_pcd.points))
        colors_list.append(np.asarray(downinst_pcd.colors))
        sem_list.append(np.full((len(downinst_pcd.points),), sem_id))
        inst_list.append(np.full((len(downinst_pcd.points),), inst))
        total_downsample_points += len(downinst_pcd.points)
        
        # Create an array to hold the downsampled normals
        downsampled_inst_normals = []
        pcd_tree = o3d.geometry.KDTreeFlann(inst_pcd)
        # Iterate over downsampled points
        for point in downinst_pcd.points:
            # Find the nearest neighbor in the original point cloud
            [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
            # Get the corresponding normal
            downsampled_inst_normals.append(inst_normals[idx[0]])
        normals_list.append(np.asarray(downsampled_inst_normals))
                
    points_dsped = np.concatenate(points_list, axis=0)
    colors_dsped = np.concatenate(colors_list, axis=0)
    normals_dsped = np.concatenate(normals_list, axis=0)
    sem_dsped = np.concatenate(sem_list, axis=0)
    inst_dsped = np.concatenate(inst_list, axis=0)
    return points_dsped, colors_dsped, normals_dsped, sem_dsped, inst_dsped

def expand_instances_and_semantics(points, instance_ids, sem_ids, radius = 0.1, ignore_index = 0):
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
    expand_dict = {
        "expand_idx_records": expand_idx_records,
        "expand_inst_records": expand_inst_records,
        "expand_sem_records": expand_sem_records,
        "expand_distances": expand_distances
    }
    return expand_dict

class SceneFun3DPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/scenefun3d",
        save_dir: str = "./data/processed/scenefun3d",
        modes: tuple = ("train", "val"),
        n_jobs: int = -1,
        ignore_index: int = 0
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        # meta data
        self.ignore_index = ignore_index
        self.create_label_database(data_dir)
        self.modes = modes
        # get scene ids
        for mode in self.modes:
            scans_file = osp.join(data_dir, splits[mode])
            scans = [line.strip() for line in open(scans_file)]
            folders = []
            for scan in scans:
                scan_folder = osp.join(data_dir, 'scans', scan)
                folders.append(scan_folder)
            self.files[mode] = natsorted(folders)

    def create_label_database(self, data_dir):
        label_database = {}
        for row_id, class_id in enumerate(CLASS_IDS):
            label_database[class_id] = {
                "name": CLASS_LABELS[row_id],
                "validation": class_id in VALID_CLASS_IDS,
            }
        self._save_yaml(
            self.save_dir / "label_database.yaml", label_database
        )
        return label_database

    def process_file(self, folderpath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            folderpath: path to the scan folder
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scan_id = osp.basename(folderpath)
        annotation_file = f"{folderpath}/{scan_id}_annotations.json"
        motion_file = f"{folderpath}/{scan_id}_motions.json"
        crop_file = f"{folderpath}/{scan_id}_crop_mask.npy"
        mesh_file = f"{folderpath}/{scan_id}_laser_scan.ply"
        filebase = {
            "filepath": folderpath,
            'raw_filepath': str(folderpath),
            "scene": scan_id,
            "mesh_file": mesh_file,
            'segment_file': annotation_file,
            'anno_file': annotation_file,
        }
        # reading both files and checking that they are fitting
        # mesh = PlyData.read(mesh_file)
        coords, features, _ = load_ply_with_normals(mesh_file)
        crop_mask = np.load(crop_file)
        annotations = json.load(open(annotation_file))
        motions = json.load(open(motion_file))
        # filter out the annotations that are not in the crop mask
        coords_original = deepcopy(coords)
        coords = coords[crop_mask]
        features = features[crop_mask]
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))
        # load articulation
        annotations_dict = {}
        annotation_list = annotations["annotations"]
        for annotation in annotation_list:
            annotation_id = annotation["annot_id"]
            indices = annotation["indices"]
            mask = np.array(indices)
            annotations_dict[annotation_id] = mask
        motions_list = motions['motions']
        motions_dict = {}
        for motion in motions_list:
            annotation_id = motion["annot_id"]
            mask = annotations_dict[annotation_id]
            mask_fullshape = np.zeros(len(coords_original), bool)
            mask_fullshape[mask] = True
            mask_fullshape_filtered = mask_fullshape[crop_mask]
            # get index of points that are in the mask
            mask_filtered = np.where(mask_fullshape_filtered)[0]
            motion_type = motion["motion_type"]
            sem_label = sem_label_to_id[motion_type]
            axis = np.array(motion["motion_dir"])
            origin = coords_original[motion["motion_origin_idx"]]
            motions_dict[annotation_id] = {
                "sem_label": sem_label,
                "axis": axis,
                "origin": origin,
                "mask": mask_filtered
            }
        semantic_gt = np.ones(len(coords), dtype=np.int32) * self.ignore_index
        instance_gt = np.ones(len(coords), dtype=np.int32) * self.ignore_index
        articulation_dict = {}
        inst_id = 1
        for annotation_id in motions_dict.keys():
            sem_label = motions_dict[annotation_id]["sem_label"]
            mask_filtered = motions_dict[annotation_id]["mask"]
            semantic_gt[mask_filtered] = sem_label
            instance_gt[mask_filtered] = inst_id
            articulation_dict[inst_id] = {
                'origin': motions_dict[annotation_id]["origin"],
                'axis': motions_dict[annotation_id]["axis"]
            }
            inst_id += 1
    
        # prepare the data for saving    
        # coords = points[:, :3]
        # colors = points[:, 3:6]
        # normals = points[:, 6:9]
        # downsample the points
        coords, colors, normals, semantic_gt, instance_gt = downsample(
            points, semantic_gt, instance_gt, voxel_size=0.02)
        segments_placeholder = np.zeros_like(semantic_gt)
        points = np.hstack((coords, colors, normals, semantic_gt[..., None], instance_gt[..., None],
                            segments_placeholder[..., None]))
        ## save the downsampled points
        processed_filepath = (
            self.save_dir / mode / f"{scan_id}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)
        
        gt_labels = semantic_gt * 1000 + instance_gt + 1
        processed_gt_filepath = (
            self.save_dir
            / "instance_gt"
            / mode
            / (scan_id + ".txt")
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_labels.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)
        
        ## save expand_dict
        # expand_dict = expand_instances_and_semantics(coords, instance_gt, semantic_gt)
        expand_dict_file = (
            self.save_dir
            / "expand_dict"
            / (scan_id + ".pkl")
        )
        # if not expand_dict_file.parent.exists():
        #     expand_dict_file.parent.mkdir(parents=True, exist_ok=True)
        filebase['expand_dict_file'] = str(expand_dict_file)
        # with open(expand_dict_file, "wb") as f:
        #     pickle.dump(expand_dict, f)
        
        # save articulation info
        articulation_file = osp.join(self.save_dir, mode, scan_id + '_articulation.h5')
        save_dict_to_h5file(articulation_dict, articulation_file)
        filebase["articulation_gt_file"] = str(articulation_file)

        filebase["color_mean"] = [
            float((colors[:, 0] / 255).mean()),
            float((colors[:, 1] / 255).mean()),
            float((colors[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((colors[:, 0] / 255) ** 2).mean()),
            float(((colors[:, 1] / 255) ** 2).mean()),
            float(((colors[:, 2] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/scenefun3d/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    @logger.catch
    def fix_bugs_in_labels(self):
        pass
    
    def joint_database(self, train_modes=["train", "val"]):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(
                self._load_yaml(self.save_dir / (mode + "_database.yaml"))
            )
        self._save_yaml(
            self.save_dir / "train_validation_database.yaml", joint_db
        )


if __name__ == "__main__":
    Fire(SceneFun3DPreprocessing)
