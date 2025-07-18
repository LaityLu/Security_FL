from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from . import Client
from ..utils import setup_logger
from ..utils.helper import get_dis_loss

logger = setup_logger()


class SemanticAttack(Client):
    def __init__(self,
                 client_id: int,
                 model: torch.nn.Module,
                 train_dataset: Dataset,
                 list_dps: list,
                 num_dps: int,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 optimizer: dict,
                 device: str,
                 stealth_rate: float,
                 scale_weight: float,
                 poison_images: dict,
                 poison_label_swap: int,
                 *args,
                 **kwargs
                 ):
        super().__init__(client_id, model, train_dataset, list_dps, num_dps,
                         local_epochs, batch_size, loss_function, optimizer, device)
        self.poison_images = poison_images
        self.poison_label_swap = poison_label_swap
        self.stealth_rate = stealth_rate
        self.scale_weight = scale_weight

    def local_train(self) -> Tuple[torch.nn.Module, float]:

        # copy the global model in the last round
        global_model = dict()
        for key, value in self.model.state_dict().items():
            global_model[key] = value.clone().detach().requires_grad_(False)

        self.model.to(self.device)

        # start training
        self.model.train()
        # store the loss peer epoch
        epoch_loss = []

        if self.acct is None:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                # poisoning data for each batch
                for batch_idx, (images, labels) in enumerate(self.train_dl):
                    for i in range(len(self.poison_images['train'])):
                        if i == len(images):
                            break
                        images[i] = self.train_dl.dataset.dataset[self.poison_images['train'][i]][0]
                        # # add gaussian noise
                        # images[i].add_(torch.FloatTensor(images[i].shape).normal_(0, 0.01))
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
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        else:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                self.optimizer.expected_batch_size = 0
                # poisoning data for each batch
                for batch_idx, (images, labels) in enumerate(self.train_dl):
                    for i in range(len(self.poison_images['train'])):
                        if i == len(images):
                            break
                        images[i] = self.train_dl.dataset[self.poison_images['train'][i]][0]
                        # # add gaussian noise
                        # images[i].add_(torch.FloatTensor(images[i].shape).normal_(0, 0.01))
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

        # scale the model weight
        for key, value in self.model.state_dict().items():
            new_value = global_model[key] + (value - global_model[key]) * self.scale_weight
            self.model.state_dict()[key].copy_(new_value)

        # return the updated model state dict and the average loss
        return self.model, sum(epoch_loss) / len(epoch_loss)

