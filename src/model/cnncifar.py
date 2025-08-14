from torch import nn
import torch.nn.functional as F


class CNNCifar(nn.Module):
    # def __init__(self, num_classes=10, *args, **kwargs):
    #     super(CNNCifar, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
    #     self.bn1 = nn.BatchNorm2d(32)
    #     self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    #     self.bn2 = nn.BatchNorm2d(64)
    #     self.fc1 = nn.Linear(64 * 8 * 8, 512)
    #     self.fc2 = nn.Linear(512, num_classes)
    #     self.dropout = nn.Dropout(0.5)
    #
    # def forward(self, x):
    #     x = self.bn1(F.max_pool2d(F.relu(self.conv1(x)), 2))
    #     x = self.bn2(F.max_pool2d(F.relu(self.conv2(x)), 2))
    #     x = x.view(-1, 64 * 8 * 8)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)
    def __init__(self, num_classes=10, num_groups=4):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups, 32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.gn2 = nn.GroupNorm(num_groups, 64)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.gn1(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = self.gn2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
