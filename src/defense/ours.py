import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from src.aggregator import average_weights
from src.utils import setup_logger
from src.utils.helper import parameters_to_vector

logger = setup_logger()


class Ours:

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
        v_global_model = parameters_to_vector(global_model).detach().cpu().numpy()
        v_client_models = [parameters_to_vector(cm).detach().cpu().numpy() for cm in client_models]

        num_clients = len(client_idxes)

        # store the distance between each client model and global model
        cos_dis = [0.0] * num_clients  # cosine distance
        euc_dis = [0.0] * num_clients  # euclidean distance
        mht_dis = [0.0] * num_clients  # manhattan distance
        for i, m_i in enumerate(v_client_models):
            # Compute the different value of cosine distance
            cosine_distance = float(
                (1 - np.dot(m_i, v_global_model) / (np.linalg.norm(m_i) * np.linalg.norm(
                    v_global_model))) ** 2)
            # Compute the different value of Manhattan distance
            manhattan_distance = float(np.linalg.norm(m_i - v_global_model, ord=1))
            # Compute the different value of Euclidean distance
            euclidean_distance = float(np.linalg.norm(m_i - v_global_model))
            cos_dis[i] += cosine_distance
            euc_dis[i] += euclidean_distance
            mht_dis[i] += manhattan_distance

        # store the difference between different client models at three distance
        cos_dd = [0.0] * num_clients
        euc_dd = [0.0] * num_clients
        mht_dd = [0.0] * num_clients
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    c_dd = np.abs(cos_dis[i] - cos_dis[j])
                    e_dd = np.abs(euc_dis[i] - euc_dis[j])
                    m_dd = np.abs(mht_dis[i] - mht_dis[j])
                    cos_dd[i] += c_dd
                    euc_dd[i] += e_dd
                    mht_dd[i] += m_dd

        # combine into a matrix
        tri_distance_wg = np.vstack([cos_dd, mht_dd, euc_dd]).T

        # Z-score
        scaler = StandardScaler()
        new_tri_distance = scaler.fit_transform(tri_distance_wg)

        # cluster
        cluster = HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
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
