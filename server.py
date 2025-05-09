import torch
import threading
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from agent import agent,byzantine,straggler
from torch.utils.data import Subset, DataLoader

class server():
    def __init__(self, tau, n, f, r, epsilon):
        self.n = n # Total number of agents
        self.f = f # Total number of byzantines
        self.r = r # Total number of stragglers
        self.epsilon = epsilon # Threshold
        self.tau = tau # The amount of old gradients we allow
        self.grads_lock = threading.Lock()

        self.agents = []
        self.threads = []
        self.grads = []
    
    def _get_train_loaders(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
        train_dataset = datasets.MNIST(root = './data', transform = transform, download = True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 4,shuffle = True)
        return train_loader

    def initialize_agents(self, epochs=1, lr=.01):
        for _ in range(self.n-self.f-self.r):
            self.agents.append(agent(self._get_train_loaders(),epochs=epochs))
        for _ in range(self.f): 
            """
           If we have stragglers, we will wait for n-r but Byzantines can send information instantly
           which gives them an advantage. Add delay maybe? Or compute a gradient with same information but
           send random gradient once it is done?
           """
            self.agents.append(byzantine(epochs=epochs,lr=lr))
        for _ in range(self.r):
            self.agents.append(straggler(epochs=epochs,lr=lr))
        return        

    def __run_agent(self, agent):
        grad = agent.train()
        self.grads_lock.acquire()
        self.grads.append(grad)
        self.grads_lock.release()

    def begin_threading(self):
        for agent in self.agents:
            t = threading.Thread(target=self.__run_agent,args=(agent,))
            self.threads.append(t)
            t.start()

        agg_grad = self.end_threading()


    def end_threading(self):
        for t in self.threads:
            t.join()
 
        aggregated_gradient = torch.stack(self.grads).sum(dim=0)
        print("Aggregated Gradient:", aggregated_gradient)
        return aggregated_gradient