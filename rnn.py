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
        self.dropout = nn.Dropout(0.1)  # 泛化效果非常好，50轮训练 从 0.9855 -> 0.9872
        self.out = nn.Linear(64, 10)  # 10个分类

    def forward(self, x):
        # 前向传播
        r_out, _ = self.rnn(x)
        r_out = self.dropout(r_out)
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
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # @see: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU Inputs: input, h_0
            images = images.view(-1, 28, 28)
            # images = images.view(-1, 56, 14)
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


# ====================== Test ======================
from sklearn.metrics import confusion_matrix
import seaborn as sns


def rnn_test():
    model.eval()    # 将模型设置为评估模式
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(-1, 28, 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
    print('Model Accuracy =:%.4f' % (correct / total))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    # while: input_size=14
    # Function took 8m 41s to execute.
    # Model Accuracy =:0.9855
    show_sample(train_dataset)
    rnn_train(50)
    rnn_test()
