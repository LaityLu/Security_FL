from typing import Dict, Tuple, List

import numpy as np
from torch.utils.data import Dataset


class UniformSampler:
    def __init__(self,
                 dataset: Dataset,
                 num_clients: int,
                 groupby: bool = False,
                 removed_idxes: List = None,
                 **kwargs):
        """
        :param groupby: ensure every client has all labels
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.groupby = groupby
        self.num_dps = len(dataset)
        self.num_dps_per_client = int(self.num_dps / self.num_clients)
        self.removed_idxes = removed_idxes

    def sample(self) -> Dict[int, Tuple[List[int], int]]:

        # initial  clients' data points idxes dictionary
        dict_clients = {}
        list_num_dps = [self.num_dps_per_client] * self.num_clients
        # initial  all data points idxes
        all_dps_idxes = [i for i in range(self.num_dps)]
        # if attack training, exclude the poisoning data points idxes
        if self.removed_idxes is not None:
            all_dps_idxes = list(set(all_dps_idxes) - set(self.removed_idxes))
        if not self.groupby:
            # just sample the idxes randomly
            for i in range(self.num_clients):
                dict_clients[i] = set(np.random.choice(all_dps_idxes, self.num_dps_per_client, replace=False))
                # prevent the data of the last client from being insufficient
                if len(all_dps_idxes) >= 2 * self.num_dps_per_client:
                    all_dps_idxes = list(set(all_dps_idxes) - dict_clients[i])
        else:
            if 'targets' not in dir(self.dataset):
                raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
            # get labels
            all_labels = self.dataset.targets
            labels = np.unique(all_labels)
            # if necessary, exclude the data points
            if self.removed_idxes is not None:
                all_labels = np.delete(all_labels, self.removed_idxes)
            all_labels = np.array(all_labels)
            all_dps_idxes = np.array(all_dps_idxes)
            # sample the idxes by labels
            for label in labels:
                label_idxes = all_dps_idxes[all_labels == label]
                np.random.shuffle(label_idxes)
                # uniformly assign
                for i in range(self.num_clients):
                    start_idx = i * self.num_dps_per_client // len(labels)
                    end_idx = (i + 1) * self.num_dps_per_client // len(labels)
                    temp_set = set(label_idxes[start_idx:end_idx])
                    if i not in dict_clients:
                        dict_clients[i] = set()
                    dict_clients[i] = dict_clients[i] | temp_set

        _dict_clients = {key: list(value) for key, value in dict_clients.items()}

        combined_dict = dict(zip(_dict_clients.keys(), zip(_dict_clients.values(), list_num_dps)))

        return combined_dict
