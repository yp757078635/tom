import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Net继承了tor.nn.Module
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)


    def forward(self,x):
        #激活函数
        x = F.relu(self.hidden(x))
        #预测函数
        x = self.predict(x)
        return x


#unsqueeze将一维数据变为二维数据
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)  #x data(tensor),shape(100,1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

#神经网络中数据必须要Variable后才可以输入
x,y = Variable(x),Variable(y)

net = Net(1,10,1)

#SGD随机梯度下降，lr学习率
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

#均方误差
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction,y)  #prediction,y位置不能变  预测值在前，真实值在后
    optimizer.zero_grad()   #梯度清零，防止梯度爆炸  （每次计算完梯度都会保存在net.parameters()中）
    loss.backward()    #反向传播
    optimizer.step()   #以学校效率 优化梯度

    if t%5 == 0:
        #plot and show learning process
        plt.cla()  # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f' %loss.data,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()  #做动态图
plt.show()
