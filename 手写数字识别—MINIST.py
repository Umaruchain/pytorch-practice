import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 调用父类的初始化函数，初始化继承自父类的属性

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # 卷积层1，输入通道数为1，输出通道数为64 (28-3+2*p=1）/s=1 + 1 = 28
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 卷积层2，输入通道数为64，输出通道数为128 图片长宽均不变，只改变深度
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            # 最大池化层 步长为2，核的大小为2 （28 - k=2 ）/s =2 +1  = 14 压缩了长度和宽度 为 14，深度为 128
        )
        # 经过卷积现在每张图 14*14*128 个特征量
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )
        # 全连接层，把14*14*128个特征量压缩至10个特征量，中间使用激活函数和 0.5概率的Dropout函数 ，丢弃部分连接，防止过拟合

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)  # 每一张图片不知道有多少行，但是列数一定是 14*14*128
        x = self.dense(x)
        return x


mean = [0.5]
std = [0.5]
# 图片仅有一层深度
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])
# 自定义一种变换，将图片变成向量，并且对特征值进行标准差变化，因为并没有原始数据，所以假定平均值和标准差

data_train = datasets.MNIST(root="./data",
                            transform=transform,
                            train=True, download=True)
# 训练集

data_test = datasets.MNIST(root="./data",
                           transform=transform,
                           train=False)
# 训练集

data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)

n_epochs = 5
model = Model()  # 创建一个网络的实例对象
f_loss = torch.nn.CrossEntropyLoss()  # 使用交叉熵作为loss函数
optimizer = torch.optim.Adam(model.parameters())  # 使用 optim 来管理参数 ，Adam自适应优化算法来初始化学习率


for epoch in range(n_epochs):  # 训练5次
    running_loss = 0
    running_correct = 0
    count = 0
    print("Epoch {}/{}".format(epoch, n_epochs))

    print("-"*10)

    for data in data_loader_train:    # 对于每一批数据
        X_train, Y_train = data
        X_train, Y_train = Variable(X_train), Variable(Y_train)

        outputs = model(X_train)

        _, pre = torch.max(outputs.data, 1)  # 每个数据的选取10个特征值中最大的一个，将对应的特征作为预测的结果返回给pre
        optimizer.zero_grad()  # 此次 偏导清零，重新计算

        loss = f_loss(outputs, Y_train)  # 损失值

        loss.backward()  # 计算参数偏导，得出下次变化方向
        optimizer.step()  # 根据偏导，学习率，更新参数

        running_correct = (pre == Y_train).sum()  # 每个batch预测正确的个数,统计一下全部数据的预测正确个数
        acc = (pre == Y_train).float().mean()  # 每个batch预测正确率，类型为variable
        print("train: {}/64/{}".format(running_correct, acc))
        count = count+1
        if count > 10:
            break

    # 每次训练完之后，我们都训练一下
    print("test")
    testing_correct = 0
    for data in data_loader_test:
        X_test, Y_test = data
        X_test, Y_test = Variable(X_test), Variable(Y_test)
        outputs = model(X_test)
        _, pre = torch.max(outputs.data, 1)  # 选取10个特征值中最大的一个，作为预测的结果
        testing_correct += (pre == Y_test).sum()  # 每个batch预测正确的个数,统计一下全部数据的预测正确个数
        acc = (pre == Y_test).float().mean()  # 每个batch预测正确率
        print("test :{}/64/{}".format(testing_correct, acc))


