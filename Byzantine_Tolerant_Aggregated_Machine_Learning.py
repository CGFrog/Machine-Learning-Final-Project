import torch
import numpy
import threading
from server import server

network = server(2, 20, 0, 0, .01)
network.initialize_agents(epochs=1,lr=.01)
network.begin_threading()
network.end_threading()