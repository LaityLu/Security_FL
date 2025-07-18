import copy
import time

import numpy as np
import torch

from .fedEraser import FedEraser
import torch.nn.functional as F

from ..aggregator import average_weights
from ..utils import setup_logger
from ..utils.helper import model_state_dict_to_traj, evaluate_model

logger = setup_logger()


class Crab(FedEraser):
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
        logger.info(f'Crab select rounds: {self.list_select_rounds}')
        logger.info(f'Crab select clients: {self.list_select_clients}')

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

        CM_list = model_state_dict_to_traj(CM_list)
        k = int(len(CM_list) * self.X_clients)

        target_GM = model_state_dict_to_traj([GM])[0]

        similarity = []
        for client in CM_list:
            cos_sim = []
            for g_module, c_module in zip(target_GM, client):
                if len(g_module.shape) > 1:
                    cos = torch.cosine_similarity(g_module, c_module)
                    cos_sim.append(torch.mean(cos).cpu().item())
            similarity.append(np.mean(cos_sim))
        logger.debug(f'The old clients in this round: {self.select_info[rd]}')
        logger.debug(f'The similarity: {similarity}')
        sel_client = np.argsort(similarity)[::-1]
        sel_client = sel_client[:k].tolist()

        sel_client_id = [self.select_info[rd][idx] for idx in sel_client]

        logger.debug(f'Round {rd} choose: {sel_client_id}')

        return sel_client_id

    def adaptive_rollback(self):
        rollback_round = self.list_select_rounds[0]
        logger.info(f'Crab roll back to round: {rollback_round}')
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
            logger.info("----- Crab Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []

            # distribution and local training
            if rd == 0:
                # the first recover round doesn't need calibration but aggregate the old client models directly
                for idx, client_id in enumerate(remaining_clients_id):
                    client = self.clients_pool[client_id]
                    client.model.load_state_dict(remaining_clients_models[idx])
                    local_models.append(client.model)
                    # aggregation
                    self.global_model = average_weights(self.global_model, local_models)
                round_loss = 0
            else:
                for client_id in remaining_clients_id:
                    client = self.clients_pool[client_id]
                    client.local_epochs = self.local_epochs
                    client.receive_global_model(self.global_model)
                    local_model, local_loss = client.local_train()
                    local_models.append(local_model)
                    local_losses.append(local_loss)
                # calibration and aggregation
                self.global_model = self.calibration_training(self.old_global_models[rd],
                                                              remaining_clients_models,
                                                              copy.deepcopy(self.global_model.state_dict()),
                                                              local_models)
                # compute the average loss in a round
                round_loss = sum(local_losses) / len(local_losses)
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
