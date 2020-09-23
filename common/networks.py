#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 10:10
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : networks.py
# @Software: PyCharm
import numpy as np
import torch
from torch import nn


class NetThreeLayer(nn.Module):
    """
    三层神经网络模型
    """

    def __init__(self, state_shape, action_shape, device):
        super(NetThreeLayer, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, np.prod(action_shape))
        )

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        return self.model(x)


class AtariNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(AtariNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(state_shape[2], 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, np.prod(action_shape))
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x._force().transpose(2, 0, 1)[None] / 255, dtype=torch.float)
        return self.model(x)


class DuelingNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(DuelingNetwork, self).__init__()
        # 公共网络
        self.common_layer = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)
        )
        # 价值函数网络
        self.value_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        # 优势函数网络
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, np.prod(action_shape))
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        common = self.common_layer(x)
        value = self.value_layer(common)
        advantage = self.advantage_layer(common)
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value
