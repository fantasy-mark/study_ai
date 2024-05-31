#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/31 9:27
# @Author  : Mark
# @File    : cnn.py
# @Desc    :
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from utils import time_cost

# Use CUDA + GradScaler + autocast + batch_size:64
# Function took 5m 23s to execute.
# Model Accuracy =:0.9912

# ====================== Lord Dataset ======================
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化梯度尺度器
scaler = GradScaler()

# 如果设置 download=True 下载失败可以按照 data/tips.txt 网址下载放入到 data/MNIST/raw/
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              download=False,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


def show_sample():
    num_of_images = 40
    for index in range(1, num_of_images + 1):
        plt.subplot(4, 10, index)
        plt.axis('off')
        plt.imshow(train_dataset.data[index], cmap='gray_r')
    plt.show()


# ====================== Define Module Net ======================


class cnn_net(nn.Module):
    def __init__(self):
        # see cnn_process_tips.png
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 卷积1
        self.pooling1 = nn.MaxPool2d(2)  # 最大池化
        self.relu1 = nn.ReLU()  # 激活

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(512, 10)  # 全连接

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.pooling1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.relu2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


model = cnn_net()
model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# ====================== Train ======================


@time_cost
def cnn_train(epoch):
    model.train()
    loss_list = []
    for e in range(epoch):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 使用自动混合精度
            with autocast('cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item()  # 累加损失

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


def cnn_test():
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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
    show_sample()
    cnn_train(30)
    cnn_test()
