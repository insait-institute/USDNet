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

def pcl_farthest_sample(point, npoint, return_idxs = False):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint)
        point = point[indices]
        if return_idxs:
            return point, indices
        return point

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]

    if return_idxs: return point, centroids.astype(np.int32)
    return point

def compute_normals(mesh):
    # Get vertex positions
    points = mesh.GetPointsAttr().Get()
    # Get face indices and counts
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()

    if not points or not face_vertex_indices or not face_vertex_counts:
        print("Mesh data is incomplete.")
        return None

    # Convert to NumPy for easier computation
    points = np.array([list(p) for p in points])
    face_vertex_indices = np.array(face_vertex_indices)
    face_vertex_counts = np.array(face_vertex_counts)

    # Initialize normals
    normals = np.zeros_like(points)

    # Compute normals per face
    index = 0
    for count in face_vertex_counts:
        if count == 3:  # Only process triangular faces
            v0, v1, v2 = face_vertex_indices[index:index + 3]
            p0, p1, p2 = points[v0], points[v1], points[v2]

            # Compute face normal
            normal = np.cross(p1 - p0, p2 - p0)
            normal = normal / np.linalg.norm(normal)  # Normalize

            # Add to vertex normals
            normals[v0] += normal
            normals[v1] += normal
            normals[v2] += normal

        index += count

    # Normalize vertex normals
    normals = np.array([n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n for n in normals])
    return normals

class USDAParser:
    def __init__(self, file_path):
        self.file_path = file_path
        
        self.stage = Usd.Stage.Open(self.file_path)
        self.points = {}
        self.colors = {}
        self.normals = {}
                
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
                # normals = self._get_normals(prim)
                colors = self._get_color(prim)
                
                prim_path = str(prim.GetPath())
                mesh_id = int(prim_path.split('_')[-1])
                
                self.points[prim_path] = points
                self.colors[prim_path] = colors
                # self.normals[prim_path] = normals
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
                    
        # get connectivity of articulation parts
        self.parse_connectivity()
                    
    def parse_connectivity(self):
        self.edges = defaultdict(set)
        self.mesh_id_to_category = {}
        part_pathes = list(self.mesh_path_to_id.keys())
        # Function to extract the numeric part of a node
        def extract_node_number(node):
            match = re.search(r'_(\d+)$', node)
            if match:
                return int(match.group(1))
            return None

        def extract_root_part_id(path):
            # Split the path into components
            components = path.split('/')
            for component in components:
                # Match the first numeric part after the root
                match = re.search(r'_(\d+)$', component)
                if match:
                    return int(match.group(1))
            return None  # Return None if no numeric ID is found
        def get_category_and_id(path):
            # Extract the last component of the path
            last_part = path.split('/')[-1]
            # Split by the underscore to separate category and ID
            if "_" in last_part:
                category, part_id = last_part.rsplit("_", 1)  # Split from the right
                return category, int(part_id)  # Convert part_id to integer
            else:
                return last_part, None  # If no underscore, return the whole as category and None for ID

        # Process the hierarchy and build edges
        for part in part_pathes:
            components = part.split('/')
            current_node = ""
            for component in components:
                if not component:  # Skip empty components (like the first /)
                    continue
                parent_node = current_node
                current_node = f"{parent_node}/{component}" if parent_node else f"/{component}"
                # Get numbers for the parent and current node
                root_part_id = extract_root_part_id(current_node)
                current_number = extract_node_number(current_node)
                if parent_node:
                    parent_number = extract_node_number(parent_node)
                    if parent_number is not None and current_number is not None:
                        self.edges[root_part_id].add((parent_number, current_number))
                category, part_id = get_category_and_id(current_node)
                if part_id is not None:
                    mesh_name = self.mesh_id_to_path[part_id]
                    mesh_name = mesh_name.split('/')[-1]
                    reconstructed_mesh_name = f"{category}_{part_id}"
                    assert mesh_name  == reconstructed_mesh_name, f"mesh_name: {mesh_name}, reconstructed_mesh_name: {reconstructed_mesh_name}"
                    self.mesh_id_to_category[part_id] = category

    def _get_pointcloud(self, mesh_prim):
        """Extracts the point cloud from a UsdGeom.Mesh primitive."""
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr()
        if points_attr.HasValue():
            return np.array(points_attr.Get(), dtype=np.float32)
        return np.array([], dtype=np.float32)
    def _get_color(self, mesh_prim):
        """Extracts the color from a UsdGeom.Mesh primitive."""
        mesh = UsdGeom.Mesh(mesh_prim)
        color_attr = mesh.GetDisplayColorAttr()
        if color_attr.HasValue():
            return np.array(color_attr.Get(), dtype=np.float32)
        return np.array([], dtype=np.float32)
    def _get_normals(self, mesh_prim):
        # """Extracts the normals from a UsdGeom.Mesh primitive."""
        # mesh = UsdGeom.Mesh(mesh_prim)
        # normals_attr = mesh.GetNormalsAttr()
        # if normals_attr.HasValue():
        #     return np.array(normals_attr.Get(), dtype=np.float32)
        # return np.array([], dtype=np.float32)
        return compute_normals(UsdGeom.Mesh(mesh_prim))

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
    def get_mesh_colors(self):
        """Returns dictionary of meshes with their colors."""
        return self.colors
    def get_mesh_normals(self):
        """Returns dictionary of meshes with their normals."""
        return self.normals

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

class SceneParser:
    def __init__(self, scene_folder, part_res = 512):
        self.scene_folder = scene_folder
        # get the file with ".usda" extension as usda file
        usda_files = [f for f in os.listdir(scene_folder) if f.endswith('.usda')]
        self.usda_file = osp.join(self.scene_folder, usda_files[0]) 
        # load the usda file
        self.usda_parser = USDAParser(self.usda_file)  
        
        # fps for point cloud sampling
        self.part_res = part_res

    def get_data(self):
        edges_dict = self.usda_parser.edges
        edges_dict_arr = {}
        parts_dict = {}
        # get all used mesh ids
        part_mesh_ids = set(edges_dict.keys())
        for mesh_id in edges_dict:
            edges_dict_arr[mesh_id] = []
            parts_dict[mesh_id] = set()
            for edge in edges_dict[mesh_id]:
                part_mesh_ids.add(edge[0])
                part_mesh_ids.add(edge[1])
                parts_dict[mesh_id].add(edge[0])
                parts_dict[mesh_id].add(edge[1])
                edges_dict_arr[mesh_id].append(list(edge))
            edges_dict_arr[mesh_id] = np.array(edges_dict_arr[mesh_id])
        # get points, colors, normals
        parts_data = {}
        mesh_id_to_path = self.usda_parser.get_mesh_id_to_path()
        points_dict = self.usda_parser.get_mesh_points()
        colors_dict = self.usda_parser.get_mesh_colors()
        # normals_dict = self.usda_parser.get_mesh_normals()
        for mesh_id in part_mesh_ids:
            mesh_path = mesh_id_to_path[mesh_id]
            assert mesh_path in points_dict, f"mesh_id: {mesh_id} not in points_dict"
            assert mesh_path in colors_dict, f"mesh_id: {mesh_id} not in colors_dict"
            # assert mesh_path in normals_dict, f"mesh_id: {mesh_id} not in normals_dict"
            
            part_points = points_dict[mesh_path]
            part_colors = colors_dict[mesh_path]
            # part_normals = normals_dict[mesh_path]
            # assert part_points.shape[0] == part_colors.shape[0] == part_normals.shape[0], f"part_points.shape: {part_points.shape}, part_colors.shape: {part_colors.shape}, part_normals.shape: {part_normals.shape}, self.folderpath: {self.scene_folder}, mesh_path: {mesh_path}"
            assert part_points.shape[0] == part_colors.shape[0], f"part_points.shape: {part_points.shape}, part_colors.shape: {part_colors.shape}, self.folderpath: {self.scene_folder}, mesh_path: {mesh_path}"
            
            # sample points
            # print(f"part_points.shape: {part_points.shape} of mesh_path: {mesh_path} of usda file: {self.usda_file}")
            
            if part_points.shape[0] > 10:
                part_points_sampled, indices_sampled = pcl_farthest_sample(part_points, self.part_res, return_idxs=True)
                part_colors_sampled = part_colors[indices_sampled]
                # concatenate points, colors, normals
                parts_data[mesh_id] = np.hstack([part_points_sampled, part_colors_sampled])
            else:
                parts_data[mesh_id] = None
        # get categories of each part
        mesh_id_to_category = self.usda_parser.mesh_id_to_category
        
        out_dict = {
            'parts_data': parts_data,
            'part_res': self.part_res,
            'edges_dict': edges_dict_arr,
            'mesh_id_to_category': mesh_id_to_category,
            'mesh_id_to_path': mesh_id_to_path,
            'parts_dict': parts_dict,
        }
        return out_dict
        
        

class Articulate3DConnectivityPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/articulate3d",
        save_dir: str = "./data/processed/articulate3d",
        modes: tuple = ("train", "val"),
        n_jobs: int = -1,
        ignore_index: int = 0,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        # meta data
        self.ignore_index = ignore_index
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
                                   part_res = 512)
        data_out_dict = scene_parser.get_data()
        ## save points
        processed_filepath = (
            self.save_dir / mode / f"{scan_id}.h5"
        )
        filebase = {
            "filepath": folderpath,
            'raw_filepath': str(folderpath),
            "scene": scan_id,
        }
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # filter out parts with less than 10 points
        part_filtered_out = []
        for part_id in data_out_dict['parts_data']:
            if data_out_dict['parts_data'][part_id] is None:
                part_filtered_out.append(part_id)
        
        # save data info to h5 file
        data_dict = {}
        for mesh_id in data_out_dict['edges_dict']:
            root_id = mesh_id
            edges = data_out_dict['edges_dict'][root_id]
            parts = data_out_dict['parts_dict'][root_id]
            parts_list = list(parts)
            parts_list_filt = [part_id for part_id in parts_list if part_id not in part_filtered_out]
            if len(parts_list_filt) < 2:
                continue
            # get edge matrix
            edge_matrix = np.zeros((len(parts_list_filt), len(parts_list_filt)), dtype=np.int32)
            for idx in range(edges.shape[0]):
                parent_id, child_id = edges[idx]
                if parent_id in part_filtered_out or child_id in part_filtered_out:
                    continue
                parent_idx = parts_list_filt.index(parent_id)
                child_idx = parts_list_filt.index(child_id)
                edge_matrix[parent_idx, child_idx] = 1
                edge_matrix[child_idx, parent_idx] = -1
            # get point cloud list and category list
            pcl_list = []
            category_list = []
            for part_id in parts_list_filt:
                part_points = data_out_dict['parts_data'][part_id]
                pcl_list.append(part_points)
                category_list.append(data_out_dict['mesh_id_to_category'][part_id])
            pcl_arr = np.array(pcl_list)
            data_dict[mesh_id] = {
                'edge_matrix': edge_matrix,
                'pcl_arr': pcl_arr,
                # 'category_list': category_list,
                'part_ids': parts_list_filt, 
            }
        save_dict_to_h5file(data_dict, processed_filepath)
        filebase["filepath"] = str(processed_filepath)
        
        # visualize connectivity in text 
        connectivity_text_file = self.save_dir / mode / f"{scan_id}_connectivity.txt"
        edges_dict = data_out_dict['edges_dict']
        mesh_id_to_path = data_out_dict['mesh_id_to_path']
        text_to_write = []
        for mesh_id in data_dict:
            text_to_write.append("====================================\n")
            text_to_write.append(f"mesh_id: {mesh_id}, mesh_path: {mesh_id_to_path[mesh_id]}\n")
            text_to_write.append("Edges:\n")
            for edge in edges_dict[mesh_id]:
                parent_id, child_id = edge
                if parent_id in part_filtered_out or child_id in part_filtered_out:
                    continue
                text_to_write.append(f"     parent_path: {mesh_id_to_path[parent_id]}, child_path: {mesh_id_to_path[child_id]}\n")
            # edge matrix to text
            text_to_write.append("Edge matrix:\n")
            matrix_str = np.array2string(data_dict[mesh_id]['edge_matrix'], separator=', ')
            text_to_write.append(matrix_str)
            text_to_write.append("\n")
            # part ids
            text_to_write.append("Part ids \n")
            text_to_write.append(','.join(str(data_dict[mesh_id]['part_ids'])))
            # # categories
            # text_to_write.append("Categories \n")
            # text_to_write.append(','.join(data_dict[mesh_id]['category_list']))
            # pcl shape
            text_to_write.append("pcl shape: \n")
            text_to_write.append(str(data_dict[mesh_id]['pcl_arr'].shape))
            
        # write to file
        with open(connectivity_text_file, 'w') as f:
            f.writelines(text_to_write)
             
        # aggregate colors of all parts
        # colors = []
        # for mesh_id, part_data in data_dict['parts_data'].items():
        #     colors.append(part_data[:, 3:6])
        # colors = np.concatenate(colors, axis=0)
        # filebase["color_mean"] = [
        #     float((colors[:, 0]).mean()),
        #     float((colors[:, 1]).mean()),
        #     float((colors[:, 2]).mean()),
        # ]
        # filebase["color_std"] = [
        #     float(((colors[:, 0]) ** 2).mean()),
        #     float(((colors[:, 1]) ** 2).mean()),
        #     float(((colors[:, 2]) ** 2).mean()),
        # ]
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/articulate3d/train_database.yaml",
    ):
        # train_database = self._load_yaml(train_database_path)
        # color_mean, color_std = [], []
        # for sample in train_database:
        #     color_std.append(sample["color_std"])
        #     color_mean.append(sample["color_mean"])

        # color_mean = np.array(color_mean).mean(axis=0)
        # color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        # feats_mean_std = {
        #     "mean": [float(each) for each in color_mean],
        #     "std": [float(each) for each in color_std],
        # }
        # self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)
        pass

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
    Fire(Articulate3DConnectivityPreprocessing)
