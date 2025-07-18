from typing import Dict, Tuple, List

import numpy as np
from torch.utils.data import Dataset


class ShardSampler:
    # each client has at most 4 types of label
    def __init__(self,
                 dataset: Dataset,
                 num_clients: int,
                 removed_idxes: List = None,
                 **kwargs):
        self.dataset = dataset
        self.num_clients = num_clients
        self.num_dps = len(dataset)
        self.num_dps_per_client = int(self.num_dps / self.num_clients)
        self.removed_idxes = removed_idxes

    def sample(self) -> Dict[int, Tuple[List[int], int]]:

        if 'targets' not in dir(self.dataset):
            raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
        # initial  clients' data points idxes dictionary
        dict_clients = {}
        list_num_dps = [self.num_dps_per_client] * self.num_clients
        # initial  all data points idxes
        all_dps_idxes = [i for i in range(self.num_dps)]
        # get labels
        labels = self.dataset.targets

        # if necessary, exclude the data points
        if self.removed_idxes is not None:
            all_dps_idxes = list(set(all_dps_idxes) - set(self.removed_idxes))
            labels = np.delete(labels, self.removed_idxes)

        # sort all_dps_idxes by label
        idxes_labels = np.vstack((all_dps_idxes, labels))
        idxes_labels = idxes_labels[:, idxes_labels[1, :].argsort()]
        all_dps_idxes = idxes_labels[0, :]
        # divide and assign, each client has 2 shards data points
        num_shards = self.num_clients * 2
        idxes_shard = [i for i in range(num_shards)]
        for i in range(self.num_clients):
            rand_set = set(np.random.choice(idxes_shard, 2, replace=False))
            temp_set = set()
            # for each shard idx
            for rand in rand_set:
                temp_set = temp_set | set(
                    all_dps_idxes[rand * self.num_dps_per_client // 2:(rand + 1) * self.num_dps_per_client // 2])
            dict_clients[i] = list(temp_set)
            # prevent the data of the last client from being insufficient
            if len(idxes_shard) >= 4:
                idxes_shard = list(set(idxes_shard) - rand_set)

        combined_dict = dict(zip(dict_clients.keys(), zip(dict_clients.values(), list_num_dps)))

        return combined_dict
