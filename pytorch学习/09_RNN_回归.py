import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32,1)

    def forward(self,x,h_state):
        # x (batch,time_step,input_size)
        # h_state (n_layers,batch,hidden_size)
        #r_out (batch,time_step,output_size)
        r_out,h_state = self.rnn(x,h_state)
        outs = []
        #全连接层的值存放在outs中
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state





rnn = RNN()


optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.MSELoss()
h_state = None

for step in range(60):
    start,end = step*np.pi,(step+1)*np.pi
    steps = np.linspace(start,end,TIME_STEP,dtype=np.float32)

    #(batch,time_step=None,inputs)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy((x_np[np.newaxis,:,np.newaxis]))   #shape(batch,time_step,input_size)
    y = torch.from_numpy((y_np[np.newaxis, :, np.newaxis]))


    prediction,h_state = rnn(x,h_state)
    h_state = h_state.data

    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.pause(0.05)
plt.ioff()  #关闭交互模式
plt.show()
