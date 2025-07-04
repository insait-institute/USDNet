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

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

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
sem_label_to_sem_id = {
    'background': 0,
    'rotation': 1,
    'translation': 2,
}
SCANNETPP_COLOR_MAP = {0: (0.0, 0.0, 0.0),
 1: (191, 246, 112),
 2: (110, 239, 148),
 255: (0.0, 0.0, 0.0),}

splits = {'train': "nvs_sem_train.txt",
          'val': "nvs_sem_val.txt",
          'test': "sem_test.txt"}
def filter_map_classes(mapping, count_thresh, count_type, mapping_type):
    mapping = mapping[mapping[count_type] >= count_thresh]
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict
def downsample(points, sem_gt, inst_gt, voxel_size=0.025):
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
        downinst_pcd = inst_pcd.voxel_down_sample(voxel_size=voxel_size)
        
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

class MultiscanPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/multiscan",
        save_dir: str = "./data/processed/multiscan",
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
        split_meta_file = osp.join(data_dir, 'meta/scans_split.csv')
        df = pd.read_csv(split_meta_file)
        split_to_scene_ids = df.groupby('split')['scanId'].apply(list).to_dict()
        for mode in self.modes:
            scans = split_to_scene_ids[mode]
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
        mesh_file = osp.join(folderpath, scan_id + '.ply')
        anno_file = osp.join(folderpath, scan_id + '.annotations.json')
        filebase = {
            "filepath": folderpath,
            'raw_filepath': str(folderpath),
            "scene": scan_id,
            "mesh_file": mesh_file,
            'segment_file': anno_file,
            'anno_file': anno_file,
        }
        # reading both files and checking that they are fitting
        mesh = PlyData.read(mesh_file)
        coords, features, _ = load_ply_with_normals(mesh_file)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))
    
        annotations = json.load(open(anno_file))
        part_annos = annotations['parts']
            
        num_vertices = coords.shape[0]
        semantic_gt = np.ones(num_vertices, dtype=np.int32) * self.ignore_index
        instance_gt = np.ones(num_vertices, dtype=np.int32) * self.ignore_index
        # Initialize a dictionary to collect partId lists for each vertex
        vertex_to_partIds = defaultdict(list)

        # Iterate over each face in the mesh
        for face in mesh['face']:
            # Extract vertex indices and partId from the face
            vertex_indices = face['vertex_indices']
            partId = face['partId']
            # Append the partId to the list for each vertex in the face
            for vertex in vertex_indices:
                vertex_to_partIds[vertex].append(partId)
        # Determine the vertex-wise label
        vertex_labels = np.ones(len(mesh['vertex']), dtype=int) * self.ignore_index
        for vertex, partIds in vertex_to_partIds.items():
            # Use Counter to find the most common partId and handle ties
            count = Counter(partIds)
            most_common = count.most_common()
            max_count = most_common[0][1]
            # Filter to keep only the labels that have the max count
            candidates = [label for label, count in most_common if count == max_count]
            # Choose the smallest label in case of ties
            vertex_labels[vertex] = min(candidates)
        for part_anno in part_annos:
            if 'articulations' not in part_anno:
                continue
            part_id = part_anno['partId']
            sem_cate = part_anno['articulations'][0]['type']
            sem_id = sem_label_to_sem_id[sem_cate]
            instance_gt[vertex_labels == part_id] = part_id
            semantic_gt[vertex_labels == part_id] = sem_id
        
        segments_placeholder = np.zeros_like(semantic_gt)
        coords = points[:, :3]
        colors = points[:, 3:6]
        normals = points[:, 6:9]
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
        
        # save articulation info
        part_articulation = {}
        for part_anno in part_annos:
            if 'articulations' not in part_anno:
                continue
            part_id = part_anno['partId']
            origin = part_anno['articulations'][0]['origin']
            axis = part_anno['articulations'][0]['axis']
            part_articulation[part_id] = {'origin': np.array(origin), 'axis': np.array(axis)}
        articulation_file = osp.join(self.save_dir, mode, scan_id + '_articulation.h5')
        save_dict_to_h5file(part_articulation, articulation_file)
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
        train_database_path: str = "./data/processed/scannetpp/train_database.yaml",
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
    Fire(MultiscanPreprocessing)
