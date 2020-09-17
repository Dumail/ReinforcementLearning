#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/13 23:45
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : Memory.py
# @Software: PyCharm
from typing import Dict

import numpy as np


class Memory(object):
    """智能体经验池"""

    def __init__(self, state_shape, size: int, batch_size: int = 32):
        """
        构造函数
        :param obs_dim: 状态空间的维度
        :param size: 经验池的容量大小
        :param batch_size: 采样大小
        """
        self.state_shape = state_shape
        self.obs_buffer = np.zeros([size, np.prod(state_shape)], dtype=np.float32)
        self.next_obs_buffer = np.zeros([size, np.prod(state_shape)], dtype=np.float32)
        self.actions_buffer = np.zeros([size], dtype=np.float32)
        self.rewards_buffer = np.zeros([size], dtype=np.float32)
        self.done_buffer = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.current_mem = 0  # 当前记忆指针
        self.size = 0  # 当前经验池大小

    def store(self, obs, action: np.ndarray, reward: float, next_obs, done: bool):
        """
        将一次经验放入经验池
        :param obs:状态
        :param action:行为
        :param reward: 奖励
        :param next_obs: 下一状态
        :param done: episode是否结束
        """
        self.obs_buffer[self.current_mem] = obs.flatten()
        self.actions_buffer[self.current_mem] = action
        self.rewards_buffer[self.current_mem] = reward
        self.next_obs_buffer[self.current_mem] = next_obs.flatten()
        self.done_buffer[self.current_mem] = done
        self.current_mem = (self.current_mem + 1) % self.max_size  # 循环大小
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        """
        采样
        :return: 字典形式的经验簇
        """
        index = np.random.choice(self.size, size=self.batch_size, replace=False)  # 随机选择多个索引
        obs = self.obs_buffer[index]
        next_obs = self.next_obs_buffer[index]
        if not isinstance(self.state_shape, np.int32):
            obs = obs.reshape(self.state_shape)
            next_obs = next_obs.reshape(self.state_shape)
        return dict(obs=obs, next_obs=next_obs,
                    actions=self.actions_buffer[index],
                    rewards=self.rewards_buffer[index],
                    done=self.done_buffer[index])

    def __len__(self):
        return self.size
