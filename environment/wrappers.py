#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 10:59
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : wrappers.py
# @Software: PyCharm
import gym


class MountainCarWrapper(gym.RewardWrapper):
    """
    改变爬山小车环境的奖励反馈，加速训练
    """

    def reward(self, reward):
        return reward + 0.5 if self.state[0] > 0.4 else reward  # 目标位置为0.6,当前位置靠近是提供额外奖励
