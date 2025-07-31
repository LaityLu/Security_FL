import copy
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from opacus import PrivacyEngine
from opacus.accountants.utils import GENERATE_EPSILONS_FUNC, get_noise_multiplier_with_fed_rdp
import src.client as attacker
from src import defense, recover

from src.client import Client
from src.utils import setup_logger
from src.aggregator import average_weights
from src.utils.helper import evaluate_model, evaluate_dba, evaluate_semantic_attack, set_random_seed

logger = setup_logger()


class FedAvg:
    """Federated Averaging Strategy class."""

    def __init__(
            self,
            global_model: torch.nn.Module,
            selector,
            train_dataset: Dataset,
            test_dataset: Dataset,
            client_data_dict: Dict[int, Tuple[List[int], int]],
            config: dict,
            **kwargs
    ):
        # set random seed
        set_random_seed(config['seed'])

        self.global_model = global_model
        self.selector = selector
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.client_data_dict = client_data_dict
        self.config = config

        # with DP
        self.privacy_engine = None
        if config['FL']['with_DP']:
            self.privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=config['FL']['num_clients'])
            self.initial_noise_multipliers = []

        self.config = config
        self.clients_pool: List[Client] = []
        self.adversary_list = []
        self.remaining_budgets = []

        # initiate clients
        self.initial_clients()

        # for recover
        self.old_global_models = []
        self.old_client_models = []
        self.privacy_costs = []
        self.remaining_budgets_per_client = []
        self.select_info = []
        self.malicious_clients = self.adversary_list
        self.train_losses = []
        self.time_cost = 0
        self.aggr_clients = None

    def initial_clients(self):
        # with DP
        if self.privacy_engine is not None:
            # get the privacy budget of all client's data
            total_budgets, budgets_per_client = self.generate_budgets()
            self.remaining_budgets.append(sum(budgets_per_client))

            # get the initial_noise_multiplier
            for target_epsilon in budgets_per_client:
                noise = get_noise_multiplier_with_fed_rdp(
                    target_epsilon=target_epsilon,
                    rounds=self.config['FL']['rounds'],
                    steps=self.config['Trainer']['local_epochs'],
                    recover_rounds=self.config['FL']['recover_rounds'],
                    recover_steps=self.config['FL']['recover_steps'],
                    sample_rate=self.config['Fed_rdp']['sample_rate'],
                    delta=self.config['Fed_rdp']['delta'],
                    delta_g=self.config['Fed_rdp']['delta_g'],
                    eta=self.config['Fed_rdp']['eta'],
                    noise_config=self.config['Fed_rdp']['noise_config']
                )
                self.initial_noise_multipliers.append(noise)
            logger.info(f"The initial noise_multiplier: {self.initial_noise_multipliers}")

            # initial the privacy accountants
            self.privacy_engine.prepare_fed_rdp(
                budgets=budgets_per_client,
                total_budgets=total_budgets,
                sample_rate=self.config['Fed_rdp']['sample_rate'],
                eta=self.config['Fed_rdp']['eta'],
                delta_g=self.config['Fed_rdp']['delta_g'],
            )

        if self.config['FL']['with_attack']:
            self.adversary_list = self.config['Attack']['adversary_list']

        # create the clients pool
        for i, (list_dps, num_dps) in self.client_data_dict.items():
            if i in self.adversary_list:
                client = getattr(attacker, self.config['Attack']['name'])(i, copy.deepcopy(self.global_model),
                                                                          self.train_dataset, list_dps, num_dps,
                                                                          adversary_list=self.adversary_list,
                                                                          **self.config['Attack']['args'])
            else:
                client = Client(i, copy.deepcopy(self.global_model), self.train_dataset, list_dps, num_dps,
                                **self.config['Trainer'])
            # with DP
            if self.privacy_engine is not None:
                client.make_private(privacy_engine=self.privacy_engine,
                                    noise_multiplier=self.initial_noise_multipliers[i],
                                    **self.config['Fed_rdp'])

            self.clients_pool.append(client)

    def generate_budgets(self) -> Tuple[List[List[float]], List[float]]:
        BoundedFunc = lambda values: [min(max(x, self.config['Fed_rdp']['budgets_setting']['min_epsilon']),
                                          self.config['Fed_rdp']['budgets_setting']['max_epsilon'])
                                      for x in values]
        budgets_per_client = BoundedFunc(
            GENERATE_EPSILONS_FUNC[self.config['Fed_rdp']['budgets_setting']['name']](
                self.config['FL']['num_clients'], self.config['Fed_rdp']['budgets_setting']['args']
            )
        )
        budgets_per_data = []
        for bd, (_, num_dps) in zip(budgets_per_client, self.client_data_dict.values()):
            budgets_per_data.append([bd] * num_dps)
        return budgets_per_data, budgets_per_client

    def eval_attack(self):
        attack_accuracy = 0
        # evaluate attack
        if self.config['Attack']['name'] == 'DBA':
            attack_accuracy, _ = evaluate_dba(
                self.test_dataset,
                self.global_model,
                **self.config['Attack']['args']
            )
        elif self.config['Attack']['name'] == 'SemanticAttack':
            attack_accuracy, _ = evaluate_semantic_attack(
                self.train_dataset,
                self.global_model,
                **self.config['Attack']['args']
            )

        return attack_accuracy

    def fl_train(self):
        MA = []
        BA = []
        for rd in range(self.config['FL']['rounds']):

            self.old_global_models.append(copy.deepcopy(self.global_model.state_dict()))

            # store the local models and loss
            local_models = []
            local_losses = []
            privacy_cost = 0

            # begin training
            logger.info("-----  Round {:3d}  -----".format(rd))

            # select the clients
            selected_clients = self.selector.select()
            logger.info('selected clients:{}'.format(selected_clients))
            self.select_info.append(selected_clients)

            old_cms = []
            remaining_budgets = []
            # distribution and local training
            for client_id in selected_clients:
                client = self.clients_pool[client_id]
                client.receive_global_model(self.global_model)
                local_model, local_loss = client.local_train()
                local_models.append(local_model)
                local_losses.append(local_loss)
                privacy_cost += client.privacy_cost_this_round

                old_cms.append(copy.deepcopy(local_model.state_dict()))
                remaining_budgets.append(client.remaining_budget)
            self.old_client_models.append(old_cms)
            self.privacy_costs.append(privacy_cost)
            self.remaining_budgets.append(self.remaining_budgets[0] - sum(self.privacy_costs))
            self.remaining_budgets_per_client.append(remaining_budgets)

            if self.config['FL']['with_DP']:
                logger.info(f"remaining_budget: {remaining_budgets}")

            if self.config['FL']['with_defense']:
                defender = getattr(defense, self.config['Defense']['name'])(**self.config['Defense']['args'],
                                                                            adversary_list=self.adversary_list)
                self.global_model = defender.exec(self.global_model,
                                                  local_models,
                                                  selected_clients)
                self.aggr_clients.append(defender.benign_clients)
            else:
                # normal aggregation
                self.global_model = average_weights(self.global_model, local_models)

            if rd == self.config['FL']['rounds'] - 1:
                self.old_global_models.append(copy.deepcopy(self.global_model.state_dict()))

            # compute the average loss in a round
            round_loss = sum(local_losses) / len(local_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            self.train_losses.append(round_loss)

            # evaluate the global model
            test_accuracy, test_loss = evaluate_model(
                dataset=self.test_dataset,
                model=self.global_model,
                loss_function=self.config['Trainer']['loss_function'],
                device='cuda'
            )
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            MA.append(round(test_accuracy.item(), 2))
            # evaluate attack
            if self.config['FL']['with_attack']:
                attack_accuracy = self.eval_attack()
                logger.info("Attack accuracy: {:.2f}%".format(attack_accuracy))
                BA.append(round(attack_accuracy.item(), 2))

        logger.debug(f'Main Accuracy:{MA}')
        logger.debug(f'Attacker Accuracy:{BA}')
        logger.debug(f'Privacy Cost:{self.privacy_costs}')
        logger.debug(f'Remaining Budget:{self.remaining_budgets}')

    def fl_recover(self):
        if self.config['FL']['with_recover']:
            if self.aggr_clients is None:
                self.aggr_clients = self.select_info
            logger.info("________Begin Recover________")
            recover_server = getattr(recover, self.config['Recover']['name'])(
                test_dataset=self.test_dataset,
                global_model=self.global_model,
                clients_pool=self.clients_pool,
                old_global_models=self.old_global_models,
                old_client_models=self.old_client_models,
                old_rounds=self.config['FL']['rounds'],
                select_info=self.select_info,
                malicious_clients=self.malicious_clients,
                recover_config=self.config['Recover']['args'],
                loss_function=self.config['Trainer']['loss_function'],
                train_losses=self.train_losses,
                privacy_costs=self.privacy_costs,
                remaining_budgets=self.remaining_budgets,
                remaining_budgets_per_client=self.remaining_budgets_per_client,
                aggr_clients=self.aggr_clients,
                dp_config=self.config['Fed_rdp']
            )
            self.global_model = recover_server.recover()

            client_id = [i for i in range(self.config['FL']['num_clients']) if i not in self.malicious_clients]
            logger.debug(f'Client id:{client_id}')
            remaining_budgets = []
            for c_id in client_id:
                remaining_budgets.append(self.clients_pool[c_id].remaining_budget)
            logger.debug(f'Remaining Budget:{remaining_budgets}')
            # evaluate attack
            if self.config['FL']['with_attack']:
                attack_accuracy = self.eval_attack()
                logger.info("Attack accuracy: {:.2f}%".format(attack_accuracy))
