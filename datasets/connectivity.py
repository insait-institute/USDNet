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

class ConnectivityDataset(Dataset):
    """Docstring for ConnectivityDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/articulate3d_connectivity_trainval",
        mode: Optional[str] = "train",
        reps_per_epoch=1,
    ):
        self.reps_per_epoch = reps_per_epoch
        # loading database files
        self.data_dir = data_dir
        self._data = []
        db_file_path = Path(data_dir) / f"{mode}_database.yaml"
        assert db_file_path.exists(), f"Database file {db_file_path} does not exist"
        self._data.extend(self._load_yaml(db_file_path))
            
    def __len__(self):
        return self.reps_per_epoch * len(self.data)
        
    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        data_dict = load_dict_from_h5file(self.data[idx]['filepath'])
        
        part_ids_list = []
        parts_idxs_list = [0]
        pcls_arr = []
        edge_matrix_list = []
        for mesh_root_id in data_dict:
            part_ids = data_dict[mesh_root_id]['part_ids']
            pcl_arr = data_dict[mesh_root_id]['pcl_arr']
            edge_matrix = data_dict[mesh_root_id]['edge_matrix']    
            # # transform -1 to 2 [0: no edge, 1: occupying , 2: belonging to]
            # edge_matrix[edge_matrix == -1] = 2
            
            part_ids_list.append(part_ids)
            last_obj_idx = parts_idxs_list[-1]
            parts_idxs_list.append(last_obj_idx + len(part_ids))
            pcls_arr.append(pcl_arr)
            edge_matrix_list.append(edge_matrix)
        pcls_arr = np.concatenate(pcls_arr, axis=0)
        assert parts_idxs_list[-1] == pcls_arr.shape[0], f"parts_idxs_list[-1]: {parts_idxs_list[-1]}, pcls_arr.shape[0]: {pcls_arr.shape[0]}"
        return {
            'part_ids_list': part_ids_list,
            'parts_idxs_list': parts_idxs_list,
            'pcls_arr': pcls_arr,
            'edge_matrix_list': edge_matrix_list
        }
    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

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

class ConnectivityCollate:
    def __init__(self) -> None:
        pass
    
    def __call__(self, batch):
        part_ids_list_batch = []
        pcls_arr_batch = []
        edge_matrix_list_batch = []
        
        parts_idxs_list_batch = []
        idxs_batch = [0]
        for sample in batch:
            part_ids_list_batch.append(sample['part_ids_list'])
            pcls_arr_batch.append(sample['pcls_arr'])
            edge_matrix_list_batch.append(sample['edge_matrix_list'])
            
            parts_idxs_list_batch.append(sample['parts_idxs_list'])
            
            num_parts_sample = 0
            for part_ids in sample['part_ids_list']:
                num_parts_sample += len(part_ids)
            last_sample_idx = idxs_batch[-1]
            idxs_batch.append(last_sample_idx + num_parts_sample)
        pcls_arr_batch = np.concatenate(pcls_arr_batch, axis=0)
        assert idxs_batch[-1] == pcls_arr_batch.shape[0], f"idxs_batch[-1]: {idxs_batch[-1]}, pcls_arr_batch.shape[0]: {pcls_arr_batch.shape[0]}"
        
        data_dict = {
            'batch_size': len(batch),
            'part_ids_list': part_ids_list_batch,
            'pcls_arr': pcls_arr_batch,
            'edge_matrix_list': edge_matrix_list_batch,
            'parts_idxs_list_batch': parts_idxs_list_batch,
            'idxs_batch': idxs_batch
        }
        return to_tensor(data_dict) 