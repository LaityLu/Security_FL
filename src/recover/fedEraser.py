import copy
import time

import torch

from .recoverBase import RecoverBase
from ..aggregator import average_weights
from ..aggregator.average import average_weights_by_state_dict
from ..utils import setup_logger
from ..utils.helper import evaluate_model

logger = setup_logger()


class FedEraser(RecoverBase):
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
            loss_function)
        self.rounds = recover_config['rounds']
        self.round_interval = recover_config.get('round_interval', 1)
        self.local_epochs = recover_config['local_epochs']

    def recover(self):
        MA = []
        round_losses = []
        # get the initial global model
        self.global_model.load_state_dict(self.old_global_models[0])
        for rd in range(0, self.rounds, self.round_interval):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models = self.remove_malicious_clients(
                self.select_info[rd],
                self.old_client_models[rd])
            # begin training
            logger.info("----- FedEraser Recover Round {:3d}  -----".format(rd))
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

    def calibration_training(self, old_global_model_state, old_cm_state, new_global_model_state, new_cm):

        return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

        assert len(old_cm_state) == len(new_cm)

        new_cm_state = []
        for cm in new_cm:
            new_cm_state.append(cm.state_dict())

        # aggregation
        old_param_update = average_weights_by_state_dict(old_cm_state)
        new_param_update = average_weights_by_state_dict(new_cm_state)

        for layer in old_global_model_state.keys():
            return_model_state[layer] = old_global_model_state[layer]
            if layer.split('.')[-1] == 'num_batches_tracked':
                continue
            old_param_update[layer] = old_param_update[layer] - old_global_model_state[layer]  # oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - new_global_model_state[layer]  # newCM - newGM_t

            step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
            step_direction = new_param_update[layer] / torch.norm(
                new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

            return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

        self.global_model.load_state_dict(return_model_state)

        return self.global_model
