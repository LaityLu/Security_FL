import numpy as np

from src.aggregator import average_weights
from src.utils import setup_logger
from src.utils.helper import parameters_to_vector

logger = setup_logger()


class Krum:
    # if num_selected_clients = 1, it's Krum.
    # if num_selected_clients = n - f, it's Multi-Krum.
    def __init__(self, adversary_list, num_selected_clients=1, *args, **kwargs):
        self.num_adv = len(adversary_list)
        self.num_selected = num_selected_clients
        self.adversary_list = adversary_list

    def exec(self,
             global_model,
             client_models,
             client_idxes,
             *args,
             **kwargs):
        logger.debug('Krum begin::')

        # flatten the model into a one-dimensional tensor
        v_client_models = [parameters_to_vector(cm).detach().cpu().numpy() for cm in client_models]

        # compute the distance between different clients
        num_clients = len(client_models)
        dist_matrix = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = float(np.linalg.norm(v_client_models[i] - v_client_models[j]) ** 2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # compute sum_dist and choose the client with minimum as benign client
        scores = []
        for i in range(num_clients):
            sorted_indices = np.argsort(dist_matrix[i])
            sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - self.num_adv)]])
            scores.append(sum_dist)
        sorted_list = np.argpartition(scores, self.num_selected)
        selected_index = sorted_list[:self.num_selected]
        non_selected_index = sorted_list[self.num_selected:]

        # store the idxes and scores of benign clients and malicious clients
        benign_clients = [client_idxes[i] for i in selected_index]
        benign_scores = [scores[i].round(2) for i in selected_index]
        adv_clients = [client_idxes[i] for i in non_selected_index]
        adv_scores = [scores[i].round(2) for i in non_selected_index]

        logger.debug("The benign clients: {},\n\t scores:{}".format(benign_clients, benign_scores))
        logger.debug("The malicious clients: {},\n\t scores:{}".format(adv_clients, adv_scores))

        # aggregation
        selected_models = [client_models[i] for i in selected_index]
        global_model = average_weights(global_model, selected_models)

        logger.debug('Krum end')

        # return the aggregated global model
        return global_model
