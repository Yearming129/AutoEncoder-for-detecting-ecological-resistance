# -*- coding: utf-8 -*-
# @Filename : 3.py

import os.path
import tifffile as tiff
import numpy as np
import torch
import torch.nn as nn
from model import AutoEncoder
from torchvision import transforms
from skimage.transform import resize

img1 = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Euc__2020_road.tif')
img2 = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_disturb_2020.tif')
img3 = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Euc_Watershed_2020.tif')
img4 = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_CLCD_2020.tif')
img5 = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Slope.tif')
img6 = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Aspect.tif')

target_shape = img1.shape
img5 = resize(img5, target_shape, preserve_range=True).astype(img1.dtype)

# img1 = np.nan_to_num(img1, nan=0)
# img2 = np.nan_to_num(img2, nan=0)
# img3 = np.nan_to_num(img3, nan=0)
# img4 = np.nan_to_num(img4, nan=1)
# img5 = np.nan_to_num(img5, nan=0)
# img6 = np.nan_to_num(img6, nan=-1)

# 影像归一化，但好像不太对
# def norm(x):
#     a = (x - np.max(x)) / (np.max(x) - np.min(x))
#     return a

print(f"img1 shape: {img1.shape}, min: {np.min(img1)}, max: {np.max(img1)}")
print(f"img2 shape: {img2.shape}, min: {np.min(img2)}, max: {np.max(img2)}")
print(f"img3 shape: {img3.shape}, min: {np.min(img3)}, max: {np.max(img3)}")
print(f"img4 shape: {img4.shape}, min: {np.min(img4)}, max: {np.max(img4)}")
print(f"img5 shape: {img5.shape}, min: {np.min(img5)}, max: {np.max(img5)}")
print(f"img6 shape: {img6.shape}, min: {np.min(img6)}, max: {np.max(img6)}")


def preprocess():
    # if img1.shape != img2.shape or img1.shape != img3.shape or img1.shape != img4.shape or img1.shape != img5.shape or img1.shape != img6.shape:
    #     raise ValueError("影像的形状不一致")
    # np.ma.masked_less()
    #
    # # 将6张影像叠加在一起
    stacked_image = np.stack((img1, img2, img3, img4, img5, img6), axis=-1)
    print(stacked_image.shape)
    print(f"Stacked image min: {np.min(stacked_image)}, max: {np.max(stacked_image)}")
    split_and_filter(stacked_image)


def split_and_filter(input_array):
    # 获取输入数组的形状
    rows, cols, channels = input_array.shape

    # 初始化一个列表，用于存储分割后的小块
    split_arrays = []

    # 定义小块的大小
    block_size = 224

    for row_start in range(0, rows, block_size):
        for col_start in range(0, cols, block_size):
            row_end = min(row_start + block_size, rows)
            col_end = min(col_start + block_size, cols)

            # 分割小块
            block = input_array[row_start:row_end, col_start:col_end, :]

            if not np.any(block == -2):
                # 提取第四个通道(CLCD)
                land = block[:, :, 3]
                land = np.expand_dims(land, axis=-1)

                # 将剩下的维度组合在一起
                block = np.stack([block[:, :, 0], block[:, :, 1], block[:, :, 2], block[:, :, 4], block[:, :, 5]],
                                 axis=-1)
                # b_max = np.max(block, axis=(0, 1))  # 计算第三维度的均值
                # b_min = np.min(block, axis=(0, 1))  # 计算第三维度的标准差

                # print(f'{b_max}, {b_min}')

                # if np.any(b_max == b_min):
                #     print(f"Skipping block due to uniform values: {row_start}-{row_end}, {col_start}-{col_end}")
                #     continue
                #
                # # 使用广播（broadcasting）将均值和标准差应用到第三维度
                # block = (block - b_min) / (b_max - b_min)
                block = np.append(block, land, axis=-1)

                data_path = r'C:\Users\MR\Desktop\AE\auto_encoder\npy'
                file_name = str(row_start) + '_' + str(col_start) + '.npy'
                np.save(os.path.join(data_path, file_name), block)
    print('done!')

# 调用预处理
preprocess()



