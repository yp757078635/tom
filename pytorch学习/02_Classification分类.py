import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        #激活函数
        x = F.relu(self.hidden(x))
        #预测函数
        x = self.predict(x)
        return x


n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)  #cat是将两个张量（tensor）拼接到一起
y = torch.cat((y0,y1),).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0)
# plt.show()

net = Net(2,10,2)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)

loss_func = torch.nn.CrossEntropyLoss()  #交叉熵

for t in range(100):
    #正向传播
    out = net(x)  #F.softmax(out)
    #计算损失
    loss = loss_func(out,y)
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    if t%2==0:
        plt.cla()
        prediciton = torch.max(F.softmax(out),1)[1]  #[0]为对应的概率，[1]为对应的分类
        pred_y = prediciton.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5,-4,'Accuracy = %.2f' %accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

