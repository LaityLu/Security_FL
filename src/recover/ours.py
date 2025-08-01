import copy
import time
from collections import Counter
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F

from opacus.accountants.utils import get_noise_multiplier_with_fed_rdp_recover
from src.aggregator import average_weights
from src.recover import FedEraser
from src.utils import setup_logger
from src.utils.helper import model_state_dict_to_traj, evaluate_model

logger = setup_logger()


class Ours(FedEraser):
    def __init__(self,
                 test_dataset,
                 global_model,
                 clients_pool,
                 old_global_models,
                 old_client_models,
                 old_rounds,
                 select_info,
                 malicious_clients,
                 recover_config,
                 loss_function,
                 train_losses,
                 privacy_costs,
                 remaining_budgets,
                 remaining_budgets_per_client,
                 aggr_clients,
                 dp_config,
                 *args,
                 **kwargs):
        super().__init__(
            test_dataset,
            global_model,
            clients_pool,
            old_global_models,
            old_client_models,
            old_rounds,
            select_info,
            malicious_clients,
            recover_config,
            loss_function)
        self.P_rounds = recover_config['select_round_ratio']
        self.X_clients = recover_config['select_client_ratio']
        self.alpha = recover_config['alpha']
        self.beta = recover_config['beta']
        self.gamma = recover_config['gamma']
        self.zeta = recover_config['zeta']
        self.train_losses = train_losses
        self.window_size = 0
        self.list_select_rounds = []
        self.list_select_clients = []
        self.privacy_costs = privacy_costs
        self.remaining_budgets = remaining_budgets
        self.remaining_budgets_per_client = remaining_budgets_per_client
        self.aggr_clients = aggr_clients
        self.dp_config = dp_config

    def recover(self):
        start_time = time.time()
        start_round = 0
        start_loss = self.train_losses[0]
        start_budget = self.remaining_budgets[0]
        for i in range(0, self.rounds):
            self.window_size += 1
            if self.train_losses[i] < start_loss * (1 - self.alpha) or \
                    self.remaining_budgets[i + 1] < start_budget * (1 - self.beta) or i == self.rounds - 1:
                logger.debug(f"train_loss: {self.train_losses[i] < start_loss * (1 - self.alpha)}")
                logger.debug(f"remaining_budgets: {self.remaining_budgets[i + 1] < start_budget * (1 - self.beta)}")
                sl_round = self.select_round(start_round,
                                             self.old_global_models[start_round: i + 2],
                                             self.privacy_costs[start_round: i + 1])
                self.list_select_rounds.extend(sl_round)
                for rd in sl_round:
                    sel_clients_id = self.select_client_in_round(rd,
                                                                 self.old_global_models[rd + 1],
                                                                 self.old_client_models[rd],
                                                                 self.remaining_budgets_per_client[rd])
                    self.list_select_clients.append(sel_clients_id)
                self.window_size = 0
                if i < self.rounds - 1:
                    start_round = i + 1
                    start_loss = self.train_losses[i + 1]
        logger.info(f'Ours select rounds: {self.list_select_rounds}')
        logger.info(f'Ours select clients: {self.list_select_clients}')

        rollback_round = self.adaptive_rollback()
        index = self.list_select_rounds.index(rollback_round)
        self.list_select_rounds = self.list_select_rounds[index:]
        self.list_select_clients = self.list_select_clients[index:]

        # Recalculate noise
        sel_clients_sequence = list(chain.from_iterable(self.list_select_clients[1:]))
        sel_counter = Counter(sel_clients_sequence)
        for mal_id in self.malicious_clients:
            del sel_counter[mal_id]
        for client_id, rounds in sel_counter.items():
            client = self.clients_pool[client_id]
            noise = get_noise_multiplier_with_fed_rdp_recover(
                target_epsilon=client.acct.budget,
                recover_rounds=rounds,
                recover_steps=self.local_epochs,
                sample_rate=self.dp_config['sample_rate'],
                delta=self.dp_config['delta'],
                delta_g=self.dp_config['delta_g'],
                eta=self.dp_config['eta'],
                noise_config=self.dp_config['noise_config'],
                history_privacy_costs=client.acct.privacy_costs,
                history_deltas=client.acct.deltas
            )
            client.prepare_recover(noise)

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

    def select_round(self, start_epoch, old_global_models_state_dict, privacy_costs):
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
        logger.debug(f"KL Divergence between global models in window: {kl_list}")
        logger.debug(f"Privacy Costs in window: {privacy_costs}")
        # Normalization
        kl_list = np.array(kl_list)
        kl_list /= kl_list.sum()
        privacy_costs = np.array(privacy_costs)
        privacy_costs /= privacy_costs.sum()
        # compute the scores
        scores = (1 - self.gamma) * kl_list + self.gamma * privacy_costs
        logger.debug(f"Final scores in window: {scores}")
        # choose
        sel_round = np.argsort(scores)[::-1]
        result = (sel_round[:k] + start_epoch).tolist()
        result.sort()
        logger.debug(f'This time choose: {result}')
        return result

    def select_client_in_round(self, rd, GM, CM_list, privacy_budgets):

        idxes = [[index for index, c_id in enumerate(self.select_info[rd])
                  if c_id == bgn_c_id] for bgn_c_id in self.aggr_clients[rd]]
        privacy_budgets_ = [privacy_budgets[i] for i in idxes]
        CM_list = [CM_list[i[0]] for i in idxes]
        CM_list = model_state_dict_to_traj(CM_list)

        target_GM = model_state_dict_to_traj([GM])[0]

        similarity = []
        for client in CM_list:
            cos_sim = []
            for g_module, c_module in zip(target_GM, client):
                if len(g_module.shape) > 1:
                    cos = torch.cosine_similarity(g_module, c_module)
                    cos_sim.append(torch.mean(cos).cpu().item())
            similarity.append(np.mean(cos_sim))
        logger.debug(f'The old clients in round {rd}: {self.select_info[rd]}')
        logger.debug(f'The aggregated clients in round {rd}: {self.aggr_clients[rd]}')
        logger.debug(f'The similarity: {similarity}')
        logger.debug(f'The privacy budgets: {privacy_budgets_}')

        # Normalization
        similarity = np.array(similarity)
        similarity /= similarity.sum()
        privacy_budgets_ = np.array(privacy_budgets_)
        privacy_budgets_ /= privacy_budgets_.sum()
        # compute the scores
        scores = (1 - self.zeta) * similarity + self.zeta * privacy_budgets_
        logger.debug(f'The final scores: {scores}')
        # choose
        k = int(len(CM_list) * self.X_clients)

        sel_client = np.argsort(scores)
        sel_client = sel_client[:k].tolist()

        sel_client_id = [self.aggr_clients[rd][idx] for idx in sel_client]

        logger.debug(f'Round {rd} choose: {sel_client_id}')

        return sel_client_id

    def adaptive_rollback(self):
        rollback_round = self.list_select_rounds[0]
        for rd in range(self.rounds):
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
        logger.info(f'Roll back to round: {rollback_round}')
        return rollback_round

    def adaptive_recover(self, old_global_models_state_dict, old_client_models_state_dict):
        MA = []
        round_losses = []
        # get the initial global model
        self.global_model.load_state_dict(old_global_models_state_dict[0])
        for rd in range(len(self.list_select_rounds)):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models = self.remove_malicious_clients(
                self.list_select_clients[rd],
                old_client_models_state_dict[rd])
            # begin training
            logger.info("----- Ours Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []
            remaining_budgets = []

            # distribution and local training
            if rd == 0:
                # the first recover round doesn't need calibration but aggregate the old client models directly
                for idx, client_id in enumerate(remaining_clients_id):
                    client = self.clients_pool[client_id]
                    client.model.load_state_dict(remaining_clients_models[idx])
                    local_models.append(client.model)
                    # aggregation
                    self.global_model = average_weights(self.global_model, local_models)
                    remaining_budgets.append(client.remaining_budget)
                round_loss = 0
            else:
                for client_id in remaining_clients_id:
                    client = self.clients_pool[client_id]
                    client.local_epochs = self.local_epochs
                    client.receive_global_model(self.global_model)
                    local_model, local_loss = client.local_train()
                    local_models.append(local_model)
                    local_losses.append(local_loss)
                    remaining_budgets.append(client.remaining_budget)
                # calibration and aggregation
                self.global_model = self.calibration_training(self.old_global_models[rd],
                                                              remaining_clients_models,
                                                              copy.deepcopy(self.global_model.state_dict()),
                                                              local_models)
                # compute the average loss in a round
                round_loss = sum(local_losses) / len(local_losses)

            logger.info(f"remaining_budget: {remaining_budgets}")
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            round_losses.append(round_loss)


            self.time_cost += time.time() - start_time

            # evaluate the global model
            test_accuracy, test_loss = evaluate_model(
                dataset=self.test_dataset,
                model=self.global_model,
                loss_function=self.loss_function,
                device='cuda'
            )
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            MA.append(round(test_accuracy.item(), 2))

        logger.info("----- The recover process end -----")
        logger.info(f"Total time cost: {self.time_cost}s")
        logger.debug(f'Main Accuracy:{MA}')

        return self.global_model
