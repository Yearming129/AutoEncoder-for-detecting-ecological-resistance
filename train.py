# -*- coding: utf-8 -*-
# @Filename : train.py

import argparse
import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from model import AutoEncoder
from dataset import Driver
import torcheval.metrics.functional.regression.r2_score as r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_path', type=str, default='result')
parser.add_argument('--data_path', type=str, default=r'C:\Users\MR\Desktop\AE\auto_encoder\npy')
args = parser.parse_args()

model = AutoEncoder(mode='train').to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
early_stopping_patience = 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每5个epoch学习率减半

train_dataset = Driver(args.data_path)
data_loader = DataLoader(train_dataset, batch_size=args.batch_size)

if __name__ == '__main__':
    best_accuracy = 0
    no_improvement_count = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        r2s = 0

        for batch, data in enumerate(data_loader):
            inputs = data
            optimizer.zero_grad()

            labels, outputs = model(inputs)

            labels,outputs = labels.view(-1),outputs.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 展平真实标签

            r2s += r2_score(outputs, labels)

            # predicted = outputs.reshape(outputs.size(0), -1)
            # labels = labels.reshape(labels.size(0), -1)
            # r2s += r2_score(predicted, labels)

            # correct += torch.sum(torch.eq(predicted, labels)).item()


        print(
            f'Epoch [{epoch + 1}/{args.epochs}] - Loss: {running_loss / len(data_loader):.4f} - R2: {r2s/ len(data_loader):.2f}')

        # 在每个step_size的倍数epoch后进行学习率衰减
        scheduler.step()

    # 保存训练结果
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(model.state_dict(), 'model.pth')
