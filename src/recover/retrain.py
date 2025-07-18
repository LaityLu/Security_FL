import copy
import time

from src.aggregator import average_weights
from src.recover.recoverBase import RecoverBase
from src.utils import setup_logger
from src.utils.helper import evaluate_model

logger = setup_logger()


class Retrain(RecoverBase):
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

    def recover(self):
        MA = []
        round_losses = []
        # get the initial global model
        self.global_model.load_state_dict(self.old_global_models[0])
        for rd in range(self.rounds):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, _ = self.remove_malicious_clients(self.select_info[rd])
            # begin training
            logger.info("----- Retrain Recover Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []
            # distribution and local training
            for client_id in remaining_clients_id:
                client = self.clients_pool[client_id]
                client.receive_global_model(self.global_model)
                local_model, local_loss = client.local_train()
                local_models.append(local_model)
                local_losses.append(local_loss)

            # aggregation
            self.global_model = average_weights(self.global_model, local_models)

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
