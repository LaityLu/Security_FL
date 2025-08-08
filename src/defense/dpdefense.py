import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from src.aggregator import average_weights
from src.utils import setup_logger
from src.utils.helper import parameters_to_vector

logger = setup_logger()


class DpDefense:

    def __init__(self,
                 adversary_list: list,
                 *args,
                 **kwargs):
        self.benign_clients = []
        self.adversary_list = adversary_list
        self.FN = 0
        self.FP = 0

    def exec(self,
             global_model,
             client_models,
             client_idxes,
             *args,
             **kwargs):
        logger.debug('Ours begin::')
        # flatten the model into a one-dimensional tensor
        v_client_models = [parameters_to_vector(cm).detach().cpu().numpy() for cm in client_models]

        num_clients = len(client_idxes)

        # Compute the element-wise sum
        m_feature = []
        for m_i in v_client_models:
            m_feature.append(sum(m_i))

        # store the difference between different client models
        m_feature_ = [0.0] * num_clients
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    s = np.abs(m_feature[i] - m_feature[j])
                    m_feature_[i] += s

        # combine into a matrix
        tri_distance_wg = np.array(m_feature_).T

        # Z-score
        scaler = StandardScaler()
        new_tri_distance = scaler.fit_transform(tri_distance_wg.reshape(-1, 1))

        # cluster
        cluster = HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=2, allow_single_cluster=True).fit(
            new_tri_distance)

        benign_clients = []
        adv_clients = client_idxes.copy()
        max_num_in_cluster = 0
        max_cluster_index = 0
        bgn_idx = []
        cluster_labels = cluster.labels_

        # all clients are benign
        if cluster_labels.max() < 0:
            for i in range(num_clients):
                benign_clients.append(client_idxes[i])
        else:
            # find clients in the largest cluster and regard them as benign_clients
            for index_cluster in range(cluster_labels.max() + 1):
                if len(cluster_labels[cluster_labels == index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(cluster_labels[cluster_labels == index_cluster])
            for i in range(num_clients):
                if cluster_labels[i] == max_cluster_index:
                    bgn_idx.append(i)
                    benign_clients.append(client_idxes[i])
                    adv_clients.remove(client_idxes[i])
        self.benign_clients = benign_clients
        logger.debug("cluster labels: {}".format(cluster_labels))
        logger.debug("The benign clients: {}".format(benign_clients))
        logger.debug("The malicious clients: {}".format(adv_clients))

        # compute FP and FN
        benign_clients_ = set(benign_clients)
        adv_clients_ = set(adv_clients)
        fn = len(benign_clients_ & set(self.adversary_list))
        self.FN += fn
        fp = len((set(client_idxes) - set(self.adversary_list)) & adv_clients_)
        self.FP += fp
        logger.debug("FN:{},\t FP:{}".format(self.FN, self.FP))

        # aggregation
        selected_models = [client_models[i] for i in bgn_idx]
        global_model = average_weights(global_model, selected_models)

        logger.debug('::Ours end')

        # return the aggregated global model
        return global_model
