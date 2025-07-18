from typing import Dict, Tuple, List
from torch.utils.data import Dataset
import numpy as np


class DirichletSampler:
    def __init__(self, dataset: Dataset,
                 num_clients: int,
                 alpha: float = 0.1,
                 removed_idxes: List = None,
                 **kwargs):
        """
        :param alpha: concentration parameters of Dirichlet distribution
        """
        self.dataset = dataset
        self.num_dps = len(dataset)
        self.num_clients = num_clients
        self.alpha = alpha
        self.removed_idxes = removed_idxes

    def sample(self) -> Dict[int, Tuple[List[int], int]]:

        # judge whether the attribute 'targets' is in the dataset
        if 'targets' not in dir(self.dataset):
            raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
        # initial  clients' data points idxes dictionary
        dict_clients = {}
        list_num_dps = [0] * self.num_clients
        # initial  all data points idxes
        all_dps_idxes = [i for i in range(self.num_dps)]
        # get labels
        all_labels = self.dataset.targets
        labels = np.unique(all_labels)
        # if necessary, exclude the data points
        if self.removed_idxes is not None:
            all_dps_idxes = list(set(all_dps_idxes) - set(self.removed_idxes))
            all_labels = np.delete(all_labels, self.removed_idxes)
        all_labels = np.array(all_labels)
        all_dps_idxes = np.array(all_dps_idxes)
        # produce the category proportion of each client
        proportions = np.random.dirichlet([self.alpha] * len(labels), self.num_clients)
        # for each label
        for c in labels:
            label_idxes = all_dps_idxes[all_labels == c]
            np.random.shuffle(label_idxes)
            proportions_c = proportions[:, c]
            proportions_c = (proportions_c / proportions_c.sum()) * len(label_idxes)
            proportions_c = proportions_c.astype(int)
            proportions_c[-1] = len(label_idxes) - proportions_c[:-1].sum()

            split_label_idxes = np.split(label_idxes, np.cumsum(proportions_c)[:-1])

            for client_idx, idxes in enumerate(split_label_idxes):
                if client_idx not in dict_clients:
                    dict_clients[client_idx] = []
                dict_clients[client_idx].extend(idxes)
        # compute data points num of each client
        for i in range(self.num_clients):
            list_num_dps[i] = len(dict_clients[i])

        combined_dict = dict(zip(dict_clients.keys(), zip(dict_clients.values(), list_num_dps)))

        return combined_dict
