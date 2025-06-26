import os, sys
import numpy as np
from random import Random
import torch
from dgl.convert import graph
from torch.utils.data import Sampler
from chemprop.rxn.get_data import get_graph_data

class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self):
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length

class GraphDataset():

    def __init__(self, data_id, split_id, filename):

        self._data_id = data_id
        self._split_id = split_id
        self._filename = filename
        self.load()


    def load(self):
        
        if int(self._data_id) in [1, 2]:
            # [rmol_dict, pmol_dict, reaction_dict] = np.load('./data/dataset_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)['data']
            # [rmol_dict, imol_dict, pmol_dict, reaction_dict] = np.load('./data/dataset_with_intermediate_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)['data']
            subset_id = int((self._data_id - int(self._data_id)+1e-5)*100)
            if subset_id == 0:
                [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
            else:
                [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
            #[rmol_dict, imol_dict, pmol_dict, reaction_dict] = np.load('../data/dataset_subset1_with_intermediate_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)['data']
            #[rmol_dict, imol_dict, pmol_dict, reaction_dict] = np.load('../data/dataset_subset1_with_intermediate_%d_%d.npz' %(self._data_id, self._split_id), allow_pickle=True)['data']
            
        elif int(self._data_id) == 2:
            # [rmol_dict, pmol_dict, reaction_dict] = np.load('./data/test_%d.npz' %self._split_id, allow_pickle=True)['data']
            [rmol_dict, imol_dict, pmol_dict, reaction_dict] = np.load('../data/test_%d.npz' %self._split_id, allow_pickle=True)['data']
        elif int(self._data_id) == 0:
            subset_id = int((self._data_id - int(self._data_id)+1e-5)*10)
            [rmol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
            imol_dict = {}
        elif int(self._data_id) == 4:
            subset_id = int((self._data_id - int(self._data_id)+1e-5)*100)
            if subset_id == 0:
                [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
            else:
                [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
                
        elif int(self._data_id) == 5:
            subset_id = int((self._data_id - int(self._data_id)+1e-5)*100)
            if subset_id == 0:
                [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
            else:
                [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
            
        elif int(self._data_id) == 6:
            [rmol_dict, imol_dict, pmol_dict, reaction_dict] = get_graph_data(self._filename)
        
        
        self.rmol_max_cnt = len(rmol_dict)
        self.imol_max_cnt = len(imol_dict)
        self.pmol_max_cnt = len(pmol_dict)
    
        self.rmol_n_node = [rmol_dict[j]['n_node'] for j in range(self.rmol_max_cnt)]
        self.rmol_n_edge = [rmol_dict[j]['n_edge'] for j in range(self.rmol_max_cnt)]
        self.rmol_node_attr = [rmol_dict[j]['node_attr'] for j in range(self.rmol_max_cnt)]
        self.rmol_edge_attr = [rmol_dict[j]['edge_attr'] for j in range(self.rmol_max_cnt)]
        self.rmol_src = [rmol_dict[j]['src'] for j in range(self.rmol_max_cnt)]
        self.rmol_dst = [rmol_dict[j]['dst'] for j in range(self.rmol_max_cnt)]
        
        if self.imol_max_cnt > 1:
            self.imol_n_node = [imol_dict[j]['n_node'] for j in range(self.imol_max_cnt)]
            self.imol_n_edge = [imol_dict[j]['n_edge'] for j in range(self.imol_max_cnt)]
            self.imol_node_attr = [imol_dict[j]['node_attr'] for j in range(self.imol_max_cnt)]
            self.imol_edge_attr = [imol_dict[j]['edge_attr'] for j in range(self.imol_max_cnt)]
            self.imol_src = [imol_dict[j]['src'] for j in range(self.imol_max_cnt)]
            self.imol_dst = [imol_dict[j]['dst'] for j in range(self.imol_max_cnt)]

        self.pmol_n_node = [pmol_dict[j]['n_node'] for j in range(self.pmol_max_cnt)]
        self.pmol_n_edge = [pmol_dict[j]['n_edge'] for j in range(self.pmol_max_cnt)]
        self.pmol_node_attr = [pmol_dict[j]['node_attr'] for j in range(self.pmol_max_cnt)]
        self.pmol_edge_attr = [pmol_dict[j]['edge_attr'] for j in range(self.pmol_max_cnt)]
        self.pmol_src = [pmol_dict[j]['src'] for j in range(self.pmol_max_cnt)]
        self.pmol_dst = [pmol_dict[j]['dst'] for j in range(self.pmol_max_cnt)]
        
        self.yld = reaction_dict['yld']
        self.rsmi = reaction_dict['rsmi']

        self.rmol_n_csum = [np.concatenate([[0], np.cumsum(self.rmol_n_node[j])]) for j in range(self.rmol_max_cnt)]
        self.rmol_e_csum = [np.concatenate([[0], np.cumsum(self.rmol_n_edge[j])]) for j in range(self.rmol_max_cnt)]

        if self.imol_max_cnt > 1:
            self.imol_n_csum = [np.concatenate([[0], np.cumsum(self.imol_n_node[j])]) for j in range(self.imol_max_cnt)]
            self.imol_e_csum = [np.concatenate([[0], np.cumsum(self.imol_n_edge[j])]) for j in range(self.imol_max_cnt)]

        self.pmol_n_csum = [np.concatenate([[0], np.cumsum(self.pmol_n_node[j])]) for j in range(self.pmol_max_cnt)]
        self.pmol_e_csum = [np.concatenate([[0], np.cumsum(self.pmol_n_edge[j])]) for j in range(self.pmol_max_cnt)]
        

    def __getitem__(self, idx):

        g1 = [graph((self.rmol_src[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]],
                     self.rmol_dst[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]]
                     ), num_nodes = self.rmol_n_node[j][idx])
              for j in range(self.rmol_max_cnt)]
              
        for j in range(self.rmol_max_cnt):
            g1[j].ndata['attr'] = torch.from_numpy(self.rmol_node_attr[j][self.rmol_n_csum[j][idx]:self.rmol_n_csum[j][idx+1]]).float()
            g1[j].edata['edge_attr'] = torch.from_numpy(self.rmol_edge_attr[j][self.rmol_e_csum[j][idx]:self.rmol_e_csum[j][idx+1]]).float()
        
        g2 = [graph((self.pmol_src[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]],
                     self.pmol_dst[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]]
                     ), num_nodes = self.pmol_n_node[j][idx])
              for j in range(self.pmol_max_cnt)]

        for j in range(self.pmol_max_cnt):
            g2[j].ndata['attr'] = torch.from_numpy(self.pmol_node_attr[j][self.pmol_n_csum[j][idx]:self.pmol_n_csum[j][idx+1]]).float()
            g2[j].edata['edge_attr'] = torch.from_numpy(self.pmol_edge_attr[j][self.pmol_e_csum[j][idx]:self.pmol_e_csum[j][idx+1]]).float()
            
        label = self.yld[idx]

        if self.imol_max_cnt > 1:
            g3 = [graph((self.imol_src[j][self.imol_e_csum[j][idx]:self.imol_e_csum[j][idx+1]],
                         self.imol_dst[j][self.imol_e_csum[j][idx]:self.imol_e_csum[j][idx+1]]
                         ), num_nodes = self.imol_n_node[j][idx])
                  for j in range(self.imol_max_cnt)]

            for j in range(self.imol_max_cnt):
                g3[j].ndata['attr'] = torch.from_numpy(self.imol_node_attr[j][self.imol_n_csum[j][idx]:self.imol_n_csum[j][idx+1]]).float()
                g3[j].edata['edge_attr'] = torch.from_numpy(self.imol_edge_attr[j][self.imol_e_csum[j][idx]:self.imol_e_csum[j][idx+1]]).float()

            return *g1, *g3, *g2, label
            
        else:
            return *g1, *g2, label
        
        
    def __len__(self):

        return self.yld.shape[0]
