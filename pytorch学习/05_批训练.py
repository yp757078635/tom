import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, #是否打乱顺序
    num_workers=0, #线程数量
)

#epoch 为迭代次数 step为每次迭代的batch个数 10/BATCH_SIZE
for epoch in range(3):
    for step, (batch_x,batch_y) in enumerate(loader):
        #training...
        print('Epoch:',epoch,'| Step',step,'| batch x:',
              batch_x.numpy(),'| batch y:',batch_y.numpy())
