# -*- coding: utf-8 -*-
# @Filename : model.py

import numpy as np
import torch
from torch.nn import Module, Linear, ReLU, Sequential, Sigmoid, Embedding
import torch.nn.functional as F


class AutoEncoder(Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        assert self.mode in ['train', 'test']
        self.embed =Sequential (
            Embedding(8, 1),
            Sigmoid()
        )
        self.encoder = Sequential(
            Linear(5, 12),
            Sigmoid(),
            Linear(12, 24),
            Sigmoid(),
            Linear(24, 12),
            Sigmoid(),
            Linear(12, 6),
            Sigmoid(),
            Linear(6, 1),
            Sigmoid()
        )
        self.decoder = Sequential(
            Linear(1, 6),
            Sigmoid(),
            Linear(6, 12),
            Sigmoid(),
            Linear(12, 24),
            Sigmoid(),
            Linear(24, 12),
            Sigmoid(),
            Linear(12, 13),
            Sigmoid()
        )

    def forward(self, image):
        #@@@@预测时注释掉，训练时需要
        # image = image.transpose(1, 3)

        # print(f"Input image shape: {image.shape}")  # 打印输入数据的形状

        # image = torch.transpose(image, 1, 3)

        # 提取 land（最后一个通道）
        land = image[:, :, :, -1]     # 形状 [8, 6, 1024]
        # land = image[:, :, :, -1].squeeze(-1)
        # land = land.to(torch.int64)-1
        # land = (land * 6).round().to(torch.int64)  # 映射到 [0, 6] 的整数
        #
        # # 修正索引范围，确保在 [0, 6] 之间(CLCD有7个维度)
        # land = torch.clamp(land, min=0, max=6)

        land = torch.LongTensor(land.numpy()).to('cuda')
        land = land.to('cuda')  # 展平为 1D 张量以传入 Embedding
        # land = F.one_hot(land,num_classes=8)
        land_onehot = F.one_hot(land,num_classes=8)
        land = self.embed(land)
        # land = land.view(image.shape[0], image.shape[1], image.shape[2], -1)  # 恢复到 [batch_size, H, W, embedding_dim]

        # 去掉最后一个通道的 image 数据

        image = image[:, :, :, :-1].to('cuda')
        labels = torch.cat((image, land_onehot), dim=-1)
        labels = labels.reshape(labels.size(0),-1,13)

        # # 确保 image 和 land 的形状匹配
        # print(f"Image shape: {image.shape}")
        # print(f"Land shape (after embedding): {land.shape}")

        # 调整形状：将多维数据展平为二维数据
        # land = land.unsqueeze(-1)  # 增加一个维度以匹配 image
        image = image * land
        image = image.to(torch.float32)
        image = image.reshape(image.size(0),-1 ,5)  # 展平为 [batch_size, features]

        # 打印展平后的数据形状
        # print(f"Flattened image shape: {image.shape}")

        encod_image = image.to('cuda')
        encod_image = self.encoder(encod_image)

        if self.mode == 'train':
            encod_image = self.decoder(encod_image)
            # encod_image = torch.transpose(encod_image, 3, 1)
            return labels.to('cuda'), encod_image
        elif self.mode == 'test':
            return encod_image
