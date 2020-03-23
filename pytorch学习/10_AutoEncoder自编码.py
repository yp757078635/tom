import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE =64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

train_data = dsets.MNIST(root='./mnist',train=True,transform=torchvision.transforms.ToTensor(),download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

# print(train_data.data.size())     [60000, 28, 28]
# print(train_data.targets.size())  [60000]

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = x.view(-1,28*28)
        b_y = x.view(-1,28*28)
        b_label = y
        encode , decode = autoencoder(b_x)
        loss = loss_func(decode , b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100 == 0:
            print("Epoch:",epoch,'| train loss: %.4f' %loss.item())

