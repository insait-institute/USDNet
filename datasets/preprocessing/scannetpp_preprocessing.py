import re, os, json
from pathlib import Path
from collections import OrderedDict
import os.path as osp
import numpy as np
import pandas as pd
import open3d as o3d
from fire import Fire
from natsort import natsorted
from loguru import logger

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

from datasets.scannetpp.scannetpp_constants import (
    CLASS_IDS, VALID_CLASS_IDS, CLASS_LABELS, INSTANCE_LABELS, LABEL2ID)

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

class ScannetppPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/scannetpp/scannetpp",
        save_dir: str = "./data/processed/scannetpp",
        modes: tuple = ("train", "val"),
        n_jobs: int = -1,
        ignore_index: int = 0
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        # meta data
        self.ignore_index = ignore_index
        self.create_label_database(data_dir)
        label_mapping = pd.read_csv(
            osp.join(data_dir, "metadata", "semantic_benchmark", "map_benchmark.csv"))
        self.label_mapping = filter_map_classes(
            label_mapping, count_thresh=0, count_type="count", mapping_type="semantic"
        )
        
        for mode in self.modes:
            split_filename = splits[mode]
            split_txt = osp.join(data_dir, "splits", split_filename)
            with open(split_txt, "r") as f:
                # read the scan names without the newline character
                scans = [line.strip() for line in f]   
            folders = []
            for scan in scans:
                scan_folder = osp.join(data_dir, "data", scan)
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
        mesh_file = osp.join(folderpath, "scans/mesh_aligned_0.05.ply")
        segment_file = osp.join(folderpath, "scans/segments.json")
        anno_file = osp.join(folderpath, "scans/segments_anno.json")
        filebase = {
            "filepath": folderpath,
            'raw_filepath': str(folderpath),
            "scene": scan_id,
            "mesh_file": mesh_file,
            'segment_file': segment_file,
            'anno_file': anno_file,
        }
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(mesh_file)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))
    
        # get segment ids and instance ids
        with open(segment_file) as f:
            segments = json.load(f)
        # load anno = (instance, groups of segments)
        with open(anno_file) as f:
            anno = json.load(f)
            
        seg_indices = np.array(segments["segIndices"], dtype=np.uint32)
        num_vertices = len(seg_indices)
        assert num_vertices == points.shape[0]
        semantic_gt = np.ones(num_vertices, dtype=np.int32) * self.ignore_index
        instance_gt = np.ones(num_vertices, dtype=np.int32) * self.ignore_index
        assigned = np.zeros(num_vertices, dtype=bool)
        for idx, instance in enumerate(anno["segGroups"]):
            label = instance["label"]
            # remap label
            instance["label"] = self.label_mapping.get(label, None)
            instance["label_index"] = LABEL2ID.get(instance["label"], self.ignore_index)
            if instance["label_index"] == self.ignore_index:
                continue
            # get all the vertices with segment index in this instance
            # and max number of labels not yet applied
            # mask = np.isin(seg_indices, instance["segments"]) & (labels_used < 3)
            mask = np.zeros(num_vertices, dtype=bool)
            mask[instance["segments"]] = True
            mask = np.logical_and(mask, ~assigned)
            size = mask.sum()
            if size == 0:
                continue
            # get semantic labels
            semantic_gt[mask] = instance["label_index"]
            assigned[mask] = True
            
            # store all valid instance (include ignored instance)
            if instance["label"] in INSTANCE_LABELS:
                instance_gt[mask] = instance["objectId"]
                
            gt_label_inspect = instance["label_index"] * 1000 + instance["objectId"] + 1
            if gt_label_inspect < 0:
                print("     instance[label_index]: {}; instance[objectId]: {}".format(["label_index"], instance["objectId"]))
                
        # downsample the points
        coords, colors, normals, sem_gt, inst_gt = downsample(
            points, semantic_gt, instance_gt)
        segments_placeholder = np.zeros_like(sem_gt)
        points = np.hstack((coords, colors, normals, sem_gt[..., None], inst_gt[..., None],
                            segments_placeholder[..., None]))
        ## save the downsampled points
        processed_filepath = (
            self.save_dir / mode / f"{scan_id}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)
        
        gt_labels = sem_gt * 1000 + inst_gt + 1
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
    Fire(ScannetppPreprocessing)
