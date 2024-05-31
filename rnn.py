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
from torch.utils.data import DataLoader

from utils import mnist_dataset

# %matplotlib inline

# 超参数定义
EPOCH = 10
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset, test_dataset, train_loader, test_loader = mnist_dataset(transform, 64)

test_x = test_dataset.test_data.type(torch.FloatTensor)[:] / 255.
test_y = test_dataset.test_labels.numpy()[:]

print(test_dataset.train_data.size())
print(test_dataset.train_labels.size())
plt.imshow(test_dataset.train_data[0].numpy(), cmap='gray')
plt.show()
# print(train_data)

# 使用Dataloader进行分批
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 定义网络
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=1, batch_first=True)
        self.out = nn.Linear(64, 10)  # 10个分类

    def forward(self, x):
        # 前向传播
        r_out, _ = self.rnn(x)
        # 选择 r_out的最后一个时间步
        out = self.out(r_out[:, -1, :])
        return out


# 设置使用GPU
cuda = torch.device('cuda')
rnn = RNN()
rnn = rnn.cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练&验证
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)
        # 前向传播
        output = rnn(b_x.cuda())
        # 损失函数
        loss = loss_func(output, b_y.cuda())
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = rnn(test_x.cuda())
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            # 计算准确率
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: {}, Step: {}, loss: {}, accuracy: {}'.format(epoch, step, loss, accuracy))

# 从测试集选出10个，进行验证
test_x = test_x.cuda()
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
print('预测数字', pred_y)
print('实际数字', test_y[:10])


if __name__ == '__main__':
    pass
