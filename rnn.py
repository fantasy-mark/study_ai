#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/31 16:16
# @Author  : Mark
# @File    : rnn.py
# @Desc    :
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn

from utils import mnist_dataset, show_sample, device, time_cost

# %matplotlib inline

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset, test_dataset, train_loader, test_loader = mnist_dataset(transform, 64)


# 定义网络
class rnn_net(nn.Module):
    def __init__(self):
        super(rnn_net, self).__init__()
        self.rnn = nn.GRU(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.out = nn.Linear(64, 10)  # 10个分类

    def forward(self, x):
        # 前向传播
        r_out, _ = self.rnn(x)
        # 选择 r_out的最后一个时间步
        out = self.out(r_out[:, -1, :])
        return out


model = rnn_net()
model.to(device=device)

# 设置使用GPU
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Epoch 9 - Training loss: 0.3819831430610182
# Function took 1m 32s to execute.
@time_cost
def rnn_train(epoch):
    model.train()
    loss_list = []
    for e in range(epoch):
        # for step, (b_x, b_y) in enumerate(train_loader):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 如果输入的tensor只有两个维度: (sequence_length, input_size)
            # 如果输入的tensor有三个维度: (sequence_length, batch_size, input_size)
            # 如果在定义 GRU 的时候，设置了 batch_first = True
            # 那么输入的tensor的三个维度: (batch_size, sequence_length, input_size)
            # images.view(-1, 28, 28) -> torch.Size([64, 28, 28])
            images = images.view(-1, 28, 28)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))
        loss_list.append(running_loss / len(train_loader))

    # 绘制损失函数随训练轮数的变化图
    plt.plot(range(1, epoch + 1), loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


# def rnn_test(test_data):
#     # 从测试集选出10个，进行验证
#     test_x = test_data.cuda()
#     test_output = model(test_x[:10].view(-1, 28, 28))
#     pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
#     print('预测数字', pred_y)
#     print('实际数字', test_y[:10])


if __name__ == '__main__':
    show_sample(train_dataset)
    rnn_train(10)
    # rnn_test(test_x)
