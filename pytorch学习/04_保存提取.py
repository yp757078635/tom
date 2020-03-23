import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

x= torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y= x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)

def save():
    net = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net.parameters(),lr=0.15)
    loss_fuc = torch.nn.MSELoss()

    for t in range(100):
        prediction = net(x)
        loss = loss_fuc(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net,'net.pkl')  #保存整个网络
    torch.save(net.state_dict(),'net_params.pkl')       #保存所有参数

    #plot result
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Net')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)



def restore_net():
    net = torch.load('net.pkl')
    prediction = net(x)
    #plt

    plt.subplot(132)
    plt.title('Net')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)

def restore_parames():
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net.load_state_dict(torch.load('net_params.pkl'))
    prediction = net(x)
    #plt

    plt.subplot(133)
    plt.title('Net')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
    plt.show()

save()
restore_net()
restore_parames()
