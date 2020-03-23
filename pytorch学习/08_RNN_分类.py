import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH=1
BATCH_SIZE=64
TIME_STEP=28     #rnn time step /image height
INPUT_SIZE=28    #rnn input size / image width
LR=0.01
DOWNLOAD_MNIST=False


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,   #将batch放在第一个维度
        )
        self.out=nn.Linear(64,10)
    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None)    #x(batch,time_step,input_size)  h_n分线程  h_c主线程
        out = self.out(r_out[:,-1,:])     #r_out(batch,time_step,input)
        return out

train_data = dsets.MNIST(root='./mnist',train=True,transform=transforms.ToTensor(),download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
test_data = dsets.MNIST(root='./mnist',train=False,transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255.       #volatile属性为True的节点不会求导
test_y = test_data.targets.squeeze()[:2000]


rnn=RNN().cuda()

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28)).cuda()
        b_y = Variable(y).cuda()

        output = rnn(b_x)

        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 ==0:

            test_output = rnn(test_x.cuda())
            print(test_output.data.size())
            #torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy()       #只返回最大值的每个索引
            accuracy = float((pred_y==test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))      #item（）作用将一个零维张量转换成浮点数
            print("Epoch:",epoch,'| train loss: %.4f' %loss.item(),'| test accuracy: %.4f' %accuracy)

test_output = rnn(test_x[:10].view(-1,28,28).cuda())
cpu_pred_y =torch.max(test_output.data.cpu(),1)[1]
pred_y = cpu_pred_y.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].cpu().numpy(),'real number')

