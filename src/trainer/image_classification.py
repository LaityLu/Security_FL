import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class ImageClassificationTrainer:
    def __init__(self,
                 local_epochs: int,
                 batch_size: int,
                 loss_function: str,
                 device,
                 **kwargs
                 ):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.loss_function = getattr(F, loss_function)
        self.device = device

    def update(self,
               model: nn.Module,
               train_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               privacy_accountant=None
               ):

        model.to(self.device)

        # start training
        model.train()
        # store the loss peer epoch
        epoch_loss = []

        if privacy_accountant is None:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss_function(output, target)
                    loss.backward()
                    optimizer.step()
                    # calculate the loss
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        else:
            for epoch in range(self.local_epochs):
                # store the loss for each batch
                batch_loss = []
                optimizer.expected_batch_size = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.expected_batch_size += len(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss_function(output, target)
                    loss.backward()
                    optimizer.step()
                    # calculate the loss
                    batch_loss.append(loss.item())

                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # return the updated model state dict and the average training loss
        return model, sum(epoch_loss) / len(epoch_loss)

    def eval(self, dataset: Dataset, model: nn.Module):

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            batch_loss = []
            correct = 0
            data_size = len(dataset)
            test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.loss_function(output, target)
                batch_loss.append(loss.item())
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            test_loss = sum(batch_loss) / len(batch_loss)
            test_accuracy = 100.00 * correct / data_size
            return test_accuracy, test_loss
