import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt



net = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)

print(net)
