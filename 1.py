# -*- coding: utf-8 -*-
# @Filename : 1.py

import tifffile as tiff
import numpy as np
import torch.utils.data.dataset
from torchvision import transforms


dis_road = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Euc__2020_road.tif')
disturb = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_disturb_2020.tif')
dis_water = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Euc_Watershed_2020.tif')
clcd = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_CLCD_2020.tif')
slope = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Slope.tif')
fluc = tiff.imread(r'C:\Users\MR\Desktop\AE\auto_encoder\Nan_Aspect.tif')

print(dis_water.shape)
# toTensor = transforms.ToTensor()
#
# disturb = toTensor(disturb).to('cuda')
# dis_road = toTensor(dis_road).to('cuda')
# dis_water = toTensor(dis_water).to('cuda')
# clcd = toTensor(clcd).to('cuda')
# slope = toTensor(slope).to('cuda')
# fluc = toTensor(fluc).to('cuda')
# a[a<0] = -1

# dis_road
# dis_water
# lu =

# a = np.ma.masked_equal(a, -3.402823e+38)
# a = np.ma.masked_invalid(a)
# print(a)

from timm.models.vision_transformer import PatchEmbed, Block


def patchify(self, imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
    # p = self.patch_embed.patch_size[0]
    # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    p = 224
    h = imgs.shape[0] // p
    w = imgs.shape[1] // p

    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def unpatchify(self, x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs
