from model import MLP
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import threading, time

class agent(object):
    def __init__(self,train_loader, epochs=1, lr = 1e-2):
        self.train_loader = train_loader
        self.epochs = epochs
        self.lr = lr

    def train(self):
        return torch.randn(100)

"""
    def train(self):
        print("Training Started")
        model = MLP(h = 100)
        optimizer = optim.SGD(model.parameters(),lr = self.lr)
        criteria = nn.NLLLoss()
        for epoch in range(self.epochs):
            running_loss = 0.0
        for images, labels in self.train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return model.parameters
    """
class byzantine(agent):
   def __init__(self,train_loader, epochs=3, lr = 1e-2):
       super(agent,self).__init__(self,train_loader, epochs, lr)

class straggler(agent):
   def __init__(self,train_loader, epochs=3, lr = 1e-2):
       super(agent,self).__init__(self,train_loader, epochs, lr)
