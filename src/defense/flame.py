import copy

import numpy as np
import torch
from sklearn.cluster import HDBSCAN

from src.utils import setup_logger
from src.utils.helper import parameters_to_vector

logger = setup_logger()


class Flame:

    def __init__(self,
                 adversary_list: list,
                 noise: float,
                 *args,
                 **kwargs):
        self.noise = noise
        self.adversary_list = adversary_list

    @torch.no_grad()
    def exec(self,
             global_model,
             client_models,
             client_idxes: list,
             *args,
             **kwargs):
        logger.debug('Flame begin::')
        num_clients = len(client_models)
        # compute the update for every client
        v_update_params = []
        for i in range(num_clients):
            v_update = []
            for old_param, new_param in zip(global_model.parameters(), client_models[i].parameters()):
                if old_param.requires_grad:
                    update = new_param.data - old_param.data
                    v_update.append(update.view(-1))
            v_update_params.append(torch.cat(v_update))
        # flatten the model into a one-dimensional tensor
        v_client_models = [parameters_to_vector(cm) for cm in client_models]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        cos_list = []
        for i in range(len(v_client_models)):
            cos_i = []
            for j in range(len(v_client_models)):
                cos_ij = 1 - cos(v_client_models[i], v_client_models[j])
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)
        cluster = HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)

        benign_client = []
        benign_updates = []
        malicious_client = client_idxes.copy()
        norm_list = np.array([])  # euclidean distance
        max_num_in_cluster = 0
        max_cluster_index = 0
        bgn_idx = []
        cluster_labels = cluster.labels_
        # all clients are benign
        if cluster_labels.max() < 0:
            for i in range(num_clients):
                benign_client.append(client_idxes[i])
                benign_updates.append(v_update_params[i])
        else:
            # find clients in the largest cluster and regard them as benign_clients
            for index_cluster in range(cluster_labels.max() + 1):
                if len(cluster_labels[cluster_labels == index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(cluster_labels[cluster_labels == index_cluster])
            for i in range(num_clients):
                if cluster_labels[i] == max_cluster_index:
                    bgn_idx.append(i)
                    benign_client.append(client_idxes[i])
                    benign_updates.append(v_update_params[i])
                    malicious_client.remove(client_idxes[i])
        for i in range(num_clients):
            norm_list = np.append(norm_list, torch.norm(v_update_params[i], p=2).item())

        logger.debug("cluster labels: {}".format(cluster_labels))
        logger.debug("The benign clients: {}".format(benign_client))
        logger.debug("The malicious clients: {}".format(malicious_client))

        clip_value = np.median(norm_list)
        for i in range(len(benign_client)):
            gama = clip_value / norm_list[i]
            if gama < 1:
                for update in benign_updates[i]:
                    update *= gama

        # aggregation and add noise
        updates = torch.zeros_like(benign_updates[0])
        for i in range(len(benign_client)):
            updates += benign_updates[i]
        updates /= len(benign_updates)
        pointer = 0
        for param in global_model.parameters():
            noise = copy.deepcopy(param.data)
            noise = noise.normal_(mean=0, std=self.noise * clip_value)
            num_param = param.numel()
            param.data += updates[pointer:pointer + num_param].view_as(param.data) + noise
            pointer += num_param

        logger.debug('Flame end::')
        return global_model
