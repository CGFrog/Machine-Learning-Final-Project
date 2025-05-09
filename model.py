import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, h = 64):
        super().__init__()
        self.fc1 = nn.Linear(28*28, h)
        self.fc2 = nn.Linear(h, 10)
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim = 1)
        return x