import re, os, json, h5py, pickle
from pathlib import Path
from collections import OrderedDict, defaultdict, Counter
import os.path as osp
import numpy as np
import pandas as pd
import open3d as o3d
from fire import Fire
from natsort import natsorted
from loguru import logger
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals
from copy import deepcopy
from pxr import Usd, UsdGeom, UsdSkel, UsdPhysics, Gf
from collections import defaultdict
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
sem_cate_to_id = {
    'background': 0,
    'rotation': 1,
    'translation': 2,
}
sem_id_to_label = {v: k for k, v in sem_cate_to_id.items()}
ARTICULATE3D_COLOR_MAP = {0: (0.0, 0.0, 0.0),
 1: (191, 246, 112),
 2: (110, 239, 148),
 255: (0.0, 0.0, 0.0),}

# splits = {'train': "train_split.txt",
#           'val': "val_split.txt"}

splits = {'train': "trainval_split.txt",
          'val': "test_split.txt"}

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

class USDAParser:
    def __init__(self, file_path, interaction_as_movement = False, exlude_stuff = False):
        self.file_path = file_path
        self.interaction_as_movement = interaction_as_movement
        self.exlude_stuff = exlude_stuff
        
        self.stage = Usd.Stage.Open(self.file_path)
        self.points = {}
                
        self.mesh_id_to_path = {}
        self.mesh_path_to_id = {}
        
        self.mesh_hierarchy = {}
        self.mesh_offspring_tree = defaultdict(list)
        self.mesh_ancestor_tree = defaultdict(None)
        
        self.is_mov = {}
        self.is_inter = {}
        self.mov_ids = []
        self.inter_ids = []
        self.parse()

    def parse(self):
        """Parses the USD file for meshes, point clouds, and joints."""
        for prim in self.stage.Traverse():
            # 1. Capture Meshes and point clouds
            if prim.IsA(UsdGeom.Mesh):
                points = self._get_pointcloud(prim)
                prim_path = str(prim.GetPath())
                mesh_id = int(prim_path.split('_')[-1])
                
                self.points[prim_path] = points
                # Extracting ID from the prim path (assuming the ID is at the end)
                
                self.mesh_id_to_path[mesh_id] = prim_path
                self.mesh_path_to_id[prim_path] = mesh_id

                # set movable and interactable flags
                self.is_mov[mesh_id] = False
                self.is_inter[mesh_id] = False
                # 2. Capture Hierarchy
                self._add_hierarchy_edge(prim_path)

        for prim in self.stage.Traverse():    
            if prim.IsA(UsdGeom.Mesh):
                # 3. Capture movable and interactable objects
                if prim.HasAttribute("movable") and bool(prim.GetAttribute("movable").Get()):
                    id = int(self.mesh_path_to_id[str(prim.GetPath())])
                    self.mov_ids.append(id)
                    self.is_mov[id] = True
                if prim.HasAttribute("interactable") and bool(prim.GetAttribute("interactable").Get()):
                    id = int(self.mesh_path_to_id[str(prim.GetPath())])
                    self.inter_ids.append(id)
                    self.is_inter[id] = True

    def _get_pointcloud(self, mesh_prim):
        """Extracts the point cloud from a UsdGeom.Mesh primitive."""
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr()
        if points_attr.HasValue():
            return np.array(points_attr.Get(), dtype=np.float32)
        return np.array([], dtype=np.float32)

    def _add_hierarchy_edge(self, prim_path_str):
                # Extract numbers using regex and store them in a list
        parts = prim_path_str.split('/')
        numbers = []
        for part in parts:
            # Find all numbers in the part
            found_numbers = re.findall(r'\d+', part)
            numbers.extend(map(int, found_numbers))  # Convert found numbers to integers
        hierarchical_info = []
        for i in range(len(numbers) - 1):
            hierarchical_info.append((numbers[i], numbers[i + 1]))
            self.mesh_offspring_tree[numbers[i]].append(numbers[i + 1])
            self.mesh_ancestor_tree[numbers[i + 1]] = numbers[i]

    def get_mesh_points(self):
        """Returns dictionary of meshes with their point clouds."""
        return self.points

    def get_mesh_id_to_path(self):
        """Returns dictionary of mesh ID to prim path mappings."""
        return self.mesh_id_to_path

    def get_mesh_hierarchy(self):
        """Returns the hierarchy structure with joint types."""
        return self.mesh_hierarchy

    def get_mov_ids(self):
        """Returns list of IDs for movable objects."""
        return self.mov_ids

    def get_inter_ids(self):
        """Returns list of IDs for interactable objects."""
        return self.inter_ids
    
    def get_mesh_ids(self):
        return self.mesh_id_to_path.keys()
    
    def get_pointcloud(self, mesh_id):
        mesh_path = self.mesh_id_to_path[mesh_id]
        mesh_ = self.meshes[mesh_path]
        # get point cloud from mesh
        pointcloud = []
        for i in range(len(mesh_)):
            pointcloud.append(list(mesh_[i]))
        pointcloud = np.array(pointcloud)
    
    def get_articulations(self):
        articulations = []
        inter_ids = self.get_inter_ids()
        for inter_id in inter_ids:
            # if inter_id in self.mov_ids:
            #     articulations.append({
            #         'interactable_id': inter_id,
            #         'movable_id': inter_id,
            #         'is_hierarchy_mov': False,
            #         'trace_list': [inter_id]
            #     })
            # else:
            # trace up to find movable object in the hierarchy
            is_hierarchy_mov = False
            mesh_mov_id = inter_id
            trace_list = [inter_id]
            hierarchy = self.mesh_ancestor_tree.get(inter_id, None)
            # exclude curtain and blind
            inter_name = str(self.mesh_id_to_path[inter_id])
            mov_name = str(self.mesh_id_to_path[mesh_mov_id])
            if 'curtain' in inter_name or 'blind' in inter_name:
                    continue
            while hierarchy is not None:
                trace_list.append(hierarchy)
                if self.is_mov[hierarchy]:
                    is_hierarchy_mov = True
                    mesh_mov_id = hierarchy
                    break
                hierarchy = self.mesh_ancestor_tree.get(hierarchy, None)
            if is_hierarchy_mov: 
                articulations.append({
                    'interactable_id': inter_id,
                    'movable_id': mesh_mov_id,
                    'is_hierarchy_mov': is_hierarchy_mov,
                    'trace_list': trace_list
                    })
            else:
                if not self.exlude_stuff:
                    articulations.append({
                        'interactable_id': inter_id,
                        'movable_id': inter_id,
                        'is_hierarchy_mov': is_hierarchy_mov,
                        'trace_list': [inter_id]
                        })
        return articulations
class SceneParser:
    def __init__(self, scene_folder, downsample_voxel_size=0.02, 
                 interaction_as_movement = False, exlude_stuff = False):
        self.scene_folder = scene_folder
        self.interaction_as_movement = interaction_as_movement
        self.exlude_stuff = exlude_stuff
        # get the file with ".usda" extension as usda file
        usda_files = [f for f in os.listdir(scene_folder) if f.endswith('.usda')]
        self.usda_file = osp.join(self.scene_folder, usda_files[0]) 
        # load the usda file
        self.usda_parser = USDAParser(self.usda_file, interaction_as_movement, exlude_stuff)
        ## get articulations from usda
        self.articulation_parts = self.usda_parser.get_articulations()
        
        # load articulation parts and params
        articulation_file = [f for f in os.listdir(scene_folder) if f.endswith('_articulation.json')][0]
        articulation_file = osp.join(scene_folder, articulation_file)
        articulation_anno = json.load(open(articulation_file, 'r'))
        self.articulation_params = {}
        for item in articulation_anno['data']['articulations']:
            pid = item['pid']
            type = item['type']
            sem_id = sem_cate_to_id[type]
            axis = item['axis']
            origin = item['origin']
            self.articulation_params[pid] = {
                'sem_id': sem_id,
                'axis': axis,
                'origin': origin
            }   
        # load mesh file
        self.mesh_file = osp.join(scene_folder, 'mesh_aligned_0.05.ply')
        self.mesh = o3d.io.read_point_cloud(self.mesh_file)
        self.mesh = self.mesh.voxel_down_sample(voxel_size=downsample_voxel_size)
        self.mesh_points = np.array(self.mesh.points)
        self.mesh_colors = (np.array(self.mesh.colors) * 255.0).astype(np.int32)
        # calculate normals 
        self.mesh.estimate_normals()
        self.mesh_normals = np.array(self.mesh.normals)
        self.mesh_kdtree = cKDTree(self.mesh_points)
        
    def get_data(self, ignore_index=0):
        num_points = self.mesh_points.shape[0]
        sem_gt = np.ones(num_points, dtype=np.int32) * ignore_index
        inst_gt = np.ones(num_points, dtype=np.int32) * ignore_index
        inter_gt = np.ones(num_points, dtype=np.int32) * ignore_index
        articulation_gt = {}
        for articulation in self.articulation_parts:
            inter_id = articulation['interactable_id']
            mov_id = articulation['movable_id']
            trace_list = articulation['trace_list']
            
            if mov_id in self.articulation_params:
                sem_id = self.articulation_params[mov_id]['sem_id']
                axis = self.articulation_params[mov_id]['axis']
                origin = self.articulation_params[mov_id]['origin']
                
                indices = self.get_subset_pcl_indexs(inter_id)
                if len(indices) != 0:
                    inter_gt[indices] = mov_id
                
                mov_mask = np.zeros(num_points, dtype=bool)
                if self.interaction_as_movement:
                    mov_mask[indices] = True
                else:
                    for trace_id in trace_list:
                        indices = self.get_subset_pcl_indexs(trace_id)
                        if len(indices) != 0:
                            mov_mask[indices] = True
                sem_gt[mov_mask] = sem_id
                inst_gt[mov_mask] = mov_id
                articulation_gt[mov_id] = {
                    'sem_id': sem_id,
                    'axis': axis,
                    'origin': origin
                }
        # get interaction mask within each movable object
        for mov_id in articulation_gt.keys():
            indices = np.where(inst_gt == mov_id)[0]
            inter_mask = inter_gt[indices] != ignore_index
            articulation_gt[mov_id]['inter_mask'] = inter_mask
        return self.mesh_points,self.mesh_colors, self.mesh_normals, \
        sem_gt, inst_gt, inter_gt, articulation_gt
    def get_subset_pcl_indexs(self, mesh_id, downsample_voxel_size=0.01, tolerance = 2e-2):
        mesh_path = self.usda_parser.mesh_id_to_path[mesh_id]
        subset_mesh = self.usda_parser.points[mesh_path]
        subset_pcl = o3d.geometry.PointCloud()
        subset_pcl.points = o3d.utility.Vector3dVector(subset_mesh)
        subset_pcl = subset_pcl.voxel_down_sample(voxel_size=downsample_voxel_size)
        subset_pcl_points = np.array(subset_pcl.points)
        
        indices = []
        for point in subset_pcl_points:
            # Find the closest point in the original point cloud
            dist, index = self.mesh_kdtree.query(point)
            # Check if the point is the same
            if dist < tolerance:
                indices.append(index)
        return np.array(list(set(indices))).astype(np.int32)    

class Articulate3DPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/articulate3d",
        save_dir: str = "./data/processed/articulate3d",
        modes: tuple = ("train", "val"),
        n_jobs: int = -1,
        ignore_index: int = 0,
        interaction_as_movement: bool = False,
        exlude_stuff: bool = False,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        # meta data
        self.ignore_index = ignore_index
        self.create_label_database(data_dir)
        self.modes = modes
        self.interaction_as_movement = interaction_as_movement
        self.exlude_stuff = exlude_stuff
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
        scene_parser = SceneParser(folderpath,
                                   interaction_as_movement=self.interaction_as_movement,
                                   exlude_stuff=self.exlude_stuff)
        coords, colors, normals, sem_gt, \
        inst_gt, inter_gt, articulation_gt = scene_parser.get_data()
        segments_placeholder = np.zeros_like(inst_gt)
        points = np.hstack([coords, colors, normals, sem_gt[..., None], inst_gt[..., None], segments_placeholder[..., None], inter_gt[..., None]])
        ## save points
        processed_filepath = (
            self.save_dir / mode / f"{scan_id}.npy"
        )
        filebase = {
            "filepath": folderpath,
            'raw_filepath': str(folderpath),
            "scene": scan_id,
        }
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
        
        # save expand_dict
        expand_dict = expand_instances_and_semantics(coords, inter_gt, sem_gt)
        expand_dict_file = (
            self.save_dir
            / "expand_dict"
            / (scan_id + ".pkl")
        )
        if not expand_dict_file.parent.exists():
            expand_dict_file.parent.mkdir(parents=True, exist_ok=True)
        filebase['expand_dict_file'] = str(expand_dict_file)
        with open(expand_dict_file, "wb") as f:
            pickle.dump(expand_dict, f)
        
        # save articulation info
        articulation_file = osp.join(self.save_dir, mode, scan_id + '_articulation.h5')
        save_dict_to_h5file(articulation_gt, articulation_file)
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
        train_database_path: str = "./data/processed/articulate3d/train_database.yaml",
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
    Fire(Articulate3DPreprocessing)
