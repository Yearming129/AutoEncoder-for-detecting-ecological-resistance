# -*- coding: utf-8 -*-
# @Filename : dataset.py

import os
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义数据集类
class Driver(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = os.listdir(data_path)  # 加载.npy文件
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = np.load(os.path.join(self.data_path,sample))
        # 在这里可以应用任何预处理或数据增强操作
        # 例如，将数据转换为张量并进行归一化
        sample = self.transform(sample)
        return sample