# -*- coding: utf-8 -*-
# @Filename : infer.py

import torch
from model import AutoEncoder

model_path = r'D:\PycharmProjects\auto_encoder\model.pth'
model = AutoEncoder(mode='test').to('cuda')

model.load_state_dict(torch.load())
with torch.no_grad():
    model.eval()

