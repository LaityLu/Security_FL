import copy
from typing import List, Tuple

import torch
from opacus import PrivacyEngine
from torch.utils.data import Dataset

from .base import Client
from ..utils import setup_logger
from ..utils.helper import get_dis_loss, add_pixel_pattern, get_noise_multiplier

logger = setup_logger()


class DBA(Client):
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
                 device: str,
                 trigger: dict,
                 poisoning_per_batch: int,
                 stealth_rate: float,
                 poison_label_swap: int,
                 adversary_list: List[int],
                 with_dp: bool,
                 *args,
                 **kwargs
                 ):
        super().__init__(client_id, model, train_dataset, list_dps, num_dps,
                         local_epochs, batch_size, loss_function, optimizer, device)
        self.trigger_args = trigger
        self.poisoning_per_batch = poisoning_per_batch
        self.stealth_rate = stealth_rate
        self.poison_label_swap = poison_label_swap
        self.adversary_list = adversary_list
        self.with_dp = with_dp

    def make_private(self,
                     privacy_engine: PrivacyEngine,
                     noise_multiplier: float,
                     noise_config: dict,
                     max_grad_norm: float,
                     max_physical_batch_size: int,
                     **kwargs):
        self.initial_noise_multiplier = noise_multiplier
        self.noise_config = noise_config
        if self.with_dp:
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
        else:
            self.acct = privacy_engine.accountant.accountants[self.id]
            self.remaining_budget = self.acct.budget

    def pretend_train_with_dp(self):
        noise_multiplier = get_noise_multiplier(self.initial_noise_multiplier, self.step,
                                                self.noise_config)
        self.step += 1
        self.acct.step(noise_multiplier, self.acct.sample_rate)

    def local_train(self) -> Tuple[torch.nn.Module, float]:
        # logger.info('DBA Client {} training'.format(self.id))
        adversarial_index = self.adversary_list.index(self.id) % 4

        # copy the global model in the last round
        global_model = dict()
        for key, value in self.model.state_dict().items():
            global_model[key] = value.clone().detach().requires_grad_(False)

        self.model.to(self.device)

        # start training
        self.model.train()
        # store the loss peer epoch
        epoch_loss = []

        if self.acct is None or not self.with_dp:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                # poisoning data for each batch
                for batch_idx, (images, labels) in enumerate(self.train_dl):
                    for i in range(self.poisoning_per_batch):
                        if i == len(images):
                            break
                        images[i] = add_pixel_pattern(images[i], adversarial_index, self.trigger_args)
                        labels[i] = self.poison_label_swap
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(images)
                    # compute classification loss
                    class_loss = self.loss_function(output, labels)
                    # compute distance loss
                    distance_loss = get_dis_loss(global_model, self.model.state_dict())
                    # compute the final loss
                    loss = (1 - self.stealth_rate) * class_loss + self.stealth_rate * distance_loss
                    loss.backward()
                    self.optimizer.step()
                    # calculate the loss
                    batch_loss.append(loss.item())
                    # print(sum(batch_loss) / len(batch_loss))
                if self.acct is not None:
                    self.pretend_train_with_dp()
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                if self.acct is not None:
                    # compute the privacy cost
                    privacy_cost = self.acct.get_epsilon(delta=0.001)
                    tempt = self.remaining_budget
                    self.remaining_budget = self.acct.budget - privacy_cost
                    self.privacy_cost_this_round = tempt - self.remaining_budget
        else:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                self.optimizer.expected_batch_size = 0
                self.change_noise()
                # poisoning data for each batch
                for batch_idx, (images, labels) in enumerate(self.train_dl):
                    for i in range(self.poisoning_per_batch):
                        if i == len(images):
                            break
                        images[i] = add_pixel_pattern(images[i], adversarial_index, self.trigger_args)
                        labels[i] = self.poison_label_swap
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.expected_batch_size += len(labels)
                    self.optimizer.zero_grad()
                    output = self.model(images)
                    # compute classification loss
                    class_loss = self.loss_function(output, labels)
                    # compute distance loss
                    distance_loss = get_dis_loss(global_model, self.model.state_dict())
                    # compute the final loss
                    loss = (1 - self.stealth_rate) * class_loss + self.stealth_rate * distance_loss
                    loss.backward()
                    self.optimizer.step()
                    # calculate the loss
                    batch_loss.append(loss.item())
                    # print(sum(batch_loss) / len(batch_loss))
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # compute the privacy cost
                privacy_cost = self.acct.get_epsilon(delta=0.001)
                tempt = self.remaining_budget
                self.remaining_budget = self.acct.budget - privacy_cost
                self.privacy_cost_this_round = tempt - self.remaining_budget

        # return the updated model state dict and the average loss
        return self.model, sum(epoch_loss) / len(epoch_loss)
