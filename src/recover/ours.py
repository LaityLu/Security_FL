import copy
import itertools
import time

import numpy as np
import torch.nn.functional as F

from src.recover import FedEraser
from src.utils import setup_logger
from src.utils.helper import model_state_dict_to_traj

logger = setup_logger()


# 假设有一个收益函数，这里用随机数代替
def gain_function(client_subset):
    # 这里应该是对模型在client_subset上的训练和评估
    # 现在只是返回一个随机收益值
    return np.random.rand()


# 计算Shapley值
def calculate_shapley_values(num_clients):
    shapley_values = np.zeros(num_clients)
    for i in range(num_clients):
        for S in itertools.chain.from_iterable(
                itertools.combinations(range(num_clients), r) for r in range(num_clients)):
            if i not in S:
                S_with_i = tuple(sorted(S + (i,)))
                marginal_contribution = gain_function(S_with_i) - gain_function(S)
                shapley_values[i] += marginal_contribution * np.math.factorial(len(S)) * np.math.factorial(
                    num_clients - len(S) - 1) / np.math.factorial(num_clients)
    return shapley_values


class Ours(FedEraser):
    def __init__(self,
                 test_dataset,
                 global_model,
                 clients_pool,
                 old_global_models,
                 old_client_models,
                 select_info,
                 malicious_clients,
                 recover_config,
                 loss_function,
                 train_losses,
                 aggr_clients,
                 *args,
                 **kwargs):
        super().__init__(
            test_dataset,
            global_model,
            clients_pool,
            old_global_models,
            old_client_models,
            select_info,
            malicious_clients,
            recover_config,
            loss_function)
        self.P_rounds = recover_config['select_round_ratio']
        self.X_clients = recover_config['select_client_ratio']
        self.alpha = recover_config['alpha']
        self.train_losses = train_losses
        self.window_size = 1
        self.list_select_rounds = []
        self.list_select_clients = []
        self.aggr_clients = aggr_clients

    def recover(self):
        start_time = time.time()
        start_round = 0
        start_loss = self.train_losses[0]
        for i in range(1, self.rounds):
            self.window_size += 1
            if self.train_losses[i] < start_loss * (1 - self.alpha) or i == self.rounds - 1:
                sl_round = self.select_round(start_round, self.old_global_models[start_round: i + 2])
                self.list_select_rounds.extend(sl_round)
                for rd in sl_round:
                    sel_clients_id = self.select_client_in_round(rd, self.old_global_models[rd + 1],
                                                                 self.old_client_models[rd])
                    self.list_select_clients.append(sel_clients_id)
                self.window_size = 0
                if i < self.rounds - 1:
                    start_round = i + 1
                    start_loss = self.train_losses[i + 1]
        logger.info(f'Our select rounds: {self.list_select_rounds}')
        logger.info(f'Our select clients: {self.list_select_clients}')

        rollback_round = self.adaptive_rollback()
        index = self.list_select_rounds.index(rollback_round)
        self.list_select_rounds = self.list_select_rounds[index:]
        self.list_select_clients = self.list_select_clients[index:]

        self.time_cost += time.time() - start_time

        sel_old_GM = []
        sel_old_CM = []
        for i, the_round in enumerate(self.list_select_rounds):
            sel_old_GM.append(self.old_global_models[the_round])
            old_CM_this_round = []
            for c_id in self.list_select_clients[i]:
                index = self.select_info[the_round].index(c_id)
                old_CM_this_round.append(self.old_client_models[the_round][index])
            sel_old_CM.append(old_CM_this_round)

        self.global_model = self.adaptive_recover(sel_old_GM, sel_old_CM)

        return self.global_model

    def select_round(self, start_epoch, old_global_models_state_dict):
        rounds = [start_epoch + i for i in range(self.window_size)]
        logger.debug(f"The rounds in window: {rounds}")
        if self.window_size == 1:
            logger.debug(f'This time choose: {[start_epoch]}')
            return [start_epoch]
        k = int(self.window_size * self.P_rounds)
        GM_trajectory = model_state_dict_to_traj(old_global_models_state_dict)
        prior = GM_trajectory[0]
        kl_list = []
        for now_traj in GM_trajectory[1:]:
            kl = 0
            for module, prior_module in zip(now_traj, prior):
                log_x = F.log_softmax(module, dim=-1)
                y = F.softmax(prior_module, dim=-1)
                kl += F.kl_div(log_x, y, reduction='sum')
            kl_list.append(kl.cpu().item())
            prior = now_traj
        logger.debug(f"KL Divergence between global models in window:\n{kl_list}")
        kl_list = np.array(kl_list)
        sel_round = np.argsort(kl_list)[::-1]
        result = (sel_round[:k] + start_epoch).tolist()
        result.sort()
        logger.debug(f'This time choose: {result}')
        return result

    def select_client_in_round(self, rd, GM, CM_list):

        idxes = [[index for index, c_id in enumerate(self.select_info[rd])
                  if c_id == bgn_c_id] for bgn_c_id in self.aggr_clients[rd]]
        CM_list = [CM_list[i[0]] for i in idxes]

        # flatten the model into a one-dimensional tensor
        v_global_model = parameters_to_vector(GM).detach().cpu().numpy()
        v_client_models = [parameters_to_vector(cm).detach().cpu().numpy() for cm in CM_list]

        num_aggr_clients = len(self.aggr_clients[rd])

        # store the distance between each client model and global model
        cos_dis_wg = [0.0] * num_aggr_clients  # cosine distance
        euc_dis_wg = [0.0] * num_aggr_clients  # euclidean distance
        mht_dis_wg = [0.0] * num_aggr_clients  # manhattan distance

        for i, m_i in enumerate(v_client_models):
            # Compute the different value of cosine distance
            cosine_distance = float(
                (1 - np.dot(m_i, v_global_model) / (np.linalg.norm(m_i) * np.linalg.norm(
                    v_global_model))) ** 2)
            # Compute the different value of Manhattan distance
            manhattan_distance = float(np.linalg.norm(m_i - v_global_model, ord=1))
            # Compute the different value of Euclidean distance
            euclidean_distance = float(np.linalg.norm(m_i - v_global_model))
            cos_dis_wg[i] += cosine_distance
            euc_dis_wg[i] += euclidean_distance
            mht_dis_wg[i] += manhattan_distance

        # compute the contributions of selected clients
        tri_distance_wg = np.vstack([cos_dis_wg, mht_dis_wg, euc_dis_wg]).T

        # decentralized
        tri_distance_wg = tri_distance_wg - tri_distance_wg.mean(axis=0)

        # compute covariance matrix and inverse matrix or pseudo-inverse matrix
        cov_matrix = np.cov(tri_distance_wg.T)
        # logger.debug('covariance matrix:{}'.format(cov_matrix))

        rank = np.linalg.matrix_rank(cov_matrix)
        if rank == cov_matrix.shape[0]:
            inv_matrix = np.linalg.inv(cov_matrix)
        else:
            inv_matrix = np.linalg.pinv(cov_matrix)

        # compute the  Mahalanobis Distance as scores
        ma_distances = []
        for i in range(num_aggr_clients):
            t = tri_distance_wg[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        logger.debug(f'The aggregated clients in this round: {self.aggr_clients[rd]}')
        logger.debug(f'The scores: {scores}')

        k = int(len(CM_list) * self.X_clients)

        sel_client = np.argsort(scores)
        sel_client = sel_client[:k].tolist()

        sel_client_id = [self.aggr_clients[rd][idx] for idx in sel_client]

        logger.debug(f'Round {rd} choose: {sel_client_id}')

        return sel_client_id

    def adaptive_rollback(self):
        rollback_round = self.list_select_rounds[0]
        for rd in range(self.old_global_round):
            if set(self.malicious_clients) & set(self.aggr_clients[rd]):
                break
        if self.list_select_rounds[0] >= rd:
            return rollback_round
        left, right = 0, len(self.list_select_rounds) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if self.list_select_rounds[mid] < rd:
                left = mid + 1
            else:
                right = mid - 1
        rollback_round = self.list_select_rounds[right]
        logger.info(f'Ours roll back to round: {rollback_round}')
        return rollback_round

    def adaptive_recover(self, local_trainer, old_global_models, old_client_models):
        local_trainer.local_epochs = self.local_epochs
        MA = []
        round_losses = []
        # get the initial global model
        new_global_model = old_global_models[0]
        for rd in range(len(self.list_select_rounds)):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models, num_dps = \
                self.remove_malicious_clients(self.list_select_clients[rd], old_client_models[rd])
            # begin training
            logger.info("----- Ours Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []

            if rd == 0:
                # the first recover round doesn't need calibration but aggregate the old cm directly
                old_cm_state = [remaining_clients_models[i].state_dict() for i in
                                range((len(remaining_clients_models)))]
                new_global_model.load_state_dict(getattr(aggregator, self.aggregator)(old_cm_state, num_dps))
                round_loss = 0
            else:
                for i, idxes in enumerate(remaining_clients_id):
                    local_model, local_loss = local_trainer.update(self.dataset_train, self.dict_clients[idxes],
                                                                   copy.deepcopy(new_global_model))
                    local_models.append(local_model)
                    local_losses.append(local_loss)

                # calibration and aggregation
                new_global_model.load_state_dict(self.calibration_training(old_global_models[rd],
                                                                           remaining_clients_models, new_global_model,
                                                                           local_models, num_dps))
                # compute the average loss in a round
                round_loss = sum(local_losses) / len(local_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            round_losses.append(round_loss)

            self.time_cost += time.time() - start_time

            # testing
            # main accuracy
            test_accuracy, test_loss = local_trainer.eval(self.dataset_test, new_global_model)
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            MA.append(round(test_accuracy.item(), 2))

        logger.info("----- The recover process end -----")
        logger.info(f"Total time cost: {self.time_cost}s")
        logger.debug(f'Main Accuracy:{MA}')

        return new_global_model.state_dict()
