

import logging
import time
import sys
import os
from tqdm.auto import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import numpy as np
from IPython.display import display
import random
import pandas as pd

class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader

def get_snpashot_idx_data_loaders(datasets: dict, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    out = {}
    for ts in tqdm(datasets.keys()):  
        dataset = CustomizedDataset(indices_list=list(range(len(datasets[ts]['edges'].src_node_ids))))

        data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
        out[ts] = data_loader
    return out


def make_data(df):

    src_node_ids = df.u.values.astype(np.long)
    dst_node_ids = df.i.values.astype(np.long)

    node_interact_times = df.ts.values.astype(np.float64)
    labels = df.label.values
    
    edge_ids = np.array(list(range(len(df.idx.values)))).astype(np.long)
    labels = df.label.values

    formatted_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                      node_interact_times=node_interact_times,
                    edge_ids=edge_ids, labels=labels)
    
    return formatted_data
    


def get_dataset_subset(graph_df,percentage):
    graph_df = graph_df.copy()

    sampled = graph_df.groupby('ts', group_keys=False).apply(lambda x: x.sample(frac=percentage))

    graph_df =  graph_df[~graph_df.isin(sampled)].dropna()
   
    return sampled,graph_df

def make_data_dictionaries(graph_dfs , d, edge_raw_features,node_raw_features ,time_varying_features,full = False):

    prev_edges = 0

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    # print(node_raw_features[1:100,:])
        
    
    for ts,graph_df in enumerate(graph_dfs.items()):
        
        graph_df = graph_df[1]

    
        if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
                edge_zero_padding = np.zeros((edge_raw_features.shape[0], 172 - edge_raw_features.shape[1]))
                edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
        if time_varying_features:
            if node_raw_features.shape[2 ] < NODE_FEAT_DIM:
                node_zero_padding = np.zeros((node_raw_features[ts].shape[0], 172 - node_raw_features[ts].shape[1]))
                node_raw_features_ts = np.concatenate([node_raw_features[ts], node_zero_padding], axis=1)
                # print(f"==>> {node_raw_features.shape=}")
                # print(f"==>> {node_zero_padding.shape=}")
        else:
          
            if node_raw_features.shape[1 ] < NODE_FEAT_DIM:
                node_zero_padding = np.zeros((node_raw_features.shape[0], 172 - node_raw_features.shape[1]))
                # print(f"==>> {node_raw_features.shape=}")
                # print(f"==>> {node_zero_padding.shape=}")
                node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
              
        assert NODE_FEAT_DIM == (node_raw_features_ts if time_varying_features else node_raw_features).shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], "Unaligned feature dimensions after feature padding!"

        data_object = make_data(graph_df)
        d[ts]= {'edges': data_object, 'node_features' : node_raw_features_ts if time_varying_features else node_raw_features  , 'edge_features':edge_raw_features}

            
        prev_edges += len(graph_df)



    return d


def get_link_prediction_data_snapshots(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    cur_edges = 0

    full_data_unstacked = make_data(graph_df)
    if len(node_raw_features.shape) > 2: 
        time_varying_features = True
    else: 
        time_varying_features = False

    
    val_graph_df , train_graph_df = get_dataset_subset(graph_df, val_ratio)


    # print(train_graph_df.ts.unique())

    test_graph_df , train_graph_df = get_dataset_subset(train_graph_df, (len(graph_df) * test_ratio ) /len(train_graph_df))
    print('loading')
    train_data_unstacked = make_data(train_graph_df)
    print('loaded')


    train_graph_dfs = dict(tuple(train_graph_df.groupby('ts')))
    val_graph_dfs = dict(tuple(val_graph_df.groupby('ts')))
    test_graph_dfs = dict(tuple(test_graph_df.groupby('ts')))
    full_data_graph_dfs =  dict(tuple(graph_df.groupby('ts')))
    del train_graph_df
    del val_graph_df
    del test_graph_df
    del graph_df    
    print('making dicts')
    

        
    train_data = make_data_dictionaries(train_graph_dfs,{},edge_raw_features,node_raw_features,time_varying_features)
    print('made train')
    del train_graph_dfs
    val_data = make_data_dictionaries(val_graph_dfs,{},edge_raw_features,node_raw_features,time_varying_features)
    del val_graph_dfs
    test_data  =make_data_dictionaries(test_graph_dfs,{},edge_raw_features,node_raw_features,time_varying_features)
    del test_graph_dfs
    full_data  =make_data_dictionaries(full_data_graph_dfs,{},edge_raw_features,node_raw_features,time_varying_features, full = True)
    del full_data_graph_dfs

    return full_data_unstacked,train_data_unstacked,train_data,val_data,test_data,full_data
  
