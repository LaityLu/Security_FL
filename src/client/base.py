import copy
from typing import List, Tuple

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from opacus import PrivacyEngine
from src.utils import DatasetSplit, setup_logger
from src.utils.helper import get_noise_multiplier

logger = setup_logger()


class Client:
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 list_dps: List,
                 num_dps: int,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 device: str
                 ):
        self.id = client_id
        self.model = model
        self.num_dps = num_dps
        self.batch_size = batch_size
        self.train_dataset = DatasetSplit(train_dataset, list_dps)
        self.train_dl = DataLoader(DatasetSplit(train_dataset, list_dps),
                                   batch_size=batch_size, shuffle=True)
        self.optimizer = getattr(optim, optimizer['name'])(self.model.parameters(),
                                                           **optimizer['args'])
        self.acct = None
        self.local_epochs = local_epochs
        self.loss_function = getattr(F, loss_function)
        self.device = device
        self.remaining_budget = 0
        self.noise_config = None
        self.initial_noise_multiplier = 0
        self.step = 0
        self.privacy_cost_this_round = 0

    def make_private(self,
                     privacy_engine: PrivacyEngine,
                     noise_multiplier: float,
                     noise_config: dict,
                     max_grad_norm: float,
                     max_physical_batch_size: int,
                     **kwargs):
        self.initial_noise_multiplier = noise_multiplier
        self.noise_config = noise_config
        self.acct = copy.deepcopy(privacy_engine.accountant.accountants[self.id])
        self.remaining_budget = self.acct.budget
        self.model, self.optimizer, self.train_dl = privacy_engine.make_private_with_fed_rdp(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_dl,
            accountant=self.acct,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            max_physical_batch_size=max_physical_batch_size
        )
        privacy_engine.accountant.accountants[self.id] = self.acct

    @torch.no_grad()
    def receive_global_model(self, global_model: torch.nn.Module):
        # update all the parameters
        for old_param, new_param in zip(self.model.parameters(), global_model.parameters()):
            old_param.data = new_param.detach().clone().to(old_param.device)

    def change_noise(self):
        noise_multiplier = get_noise_multiplier(self.initial_noise_multiplier, self.step,
                                                self.noise_config)
        self.optimizer.noise_multiplier = noise_multiplier
        self.step += 1

    def prepare_recover(self, initial_noise_multiplier, noise_config):
        self.initial_noise_multiplier = initial_noise_multiplier
        self.noise_config = noise_config
        self.step = 0

    def local_train(self) -> Tuple[torch.nn.Module, float]:
        # logger.info('Client {} training'.format(self.id))

        self.model.to(self.device)

        # start training
        self.model.train()
        # store the loss peer epoch
        epoch_loss = []
        if self.acct is None:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                for data, target in self.train_dl:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss_function(output, target)
                    loss.backward()
                    self.optimizer.step()
                    # calculate the loss
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        else:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                self.optimizer.expected_batch_size = 0
                self.change_noise()
                for batch_idx, (data, target) in enumerate(self.train_dl):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.expected_batch_size += len(target)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss_function(output, target)
                    loss.backward()
                    self.optimizer.step()
                    # calculate the loss
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # compute the privacy cost
            privacy_cost = self.acct.get_epsilon(delta=0.001)
            tempt = self.remaining_budget
            self.remaining_budget = self.acct.budget - privacy_cost
            self.privacy_cost_this_round = max(tempt - self.remaining_budget, 0)

            # return the updated model state dict and the average training loss
            return self.model, sum(epoch_loss) / len(epoch_loss)

    def get_loss(self):
        """
        It is an interference used in client side to get the loss of the model
        """
        self.model.eval()
        self.model.to(self.device)
        test_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

        loss_list = []
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_function(output, target, reduction='none')

            loss_list.extend(loss.tolist())

        return loss_list
