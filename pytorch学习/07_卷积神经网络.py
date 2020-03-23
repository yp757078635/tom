import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1,28,28)
                in_channels=1,  # 通道数
                out_channels=16,  # filter 个数
                kernel_size=5,  # filter 大小
                stride=1,  # 步长
                padding=2  # if stride = 1, padding = (kernel_size)/2
            ),  # ->(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # ->(16,14,14)
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # ->(32,14,14)
            nn.ReLU(),  # ->(32,14,14)
            nn.MaxPool2d(2)  # ->(32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        x = x.view(x.size(0), -1)  # (batch,32*7*7)    展平，拉成一维度
        output = self.out(x)
        return output


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),   #将数据变为tensor形式,压缩到255->(0,1)
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]



cnn = CNN().to(device)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = x.to(device)
        b_y = y.to(device)
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 ==0:
            test_output = cnn(test_x.to(device))
            pred_y = torch.max(test_output.cpu(),1)[1].data.numpy()
            accuracy = float((pred_y==test_y.numpy()).astype(int).sum())/float(test_y.size(0))       #item（）作用将一个零维张量转换成浮点数
            print("Epoch:",epoch,'| train loss: %.4f' %loss.item(),'| test accuracy: %.4f' %accuracy)
#
test_output = cnn(test_x[:10].to(device))
cpu_pred_y =torch.max(test_output.data.cpu(),1)[1]
pred_y = cpu_pred_y.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].cpu().numpy(),'real number')
