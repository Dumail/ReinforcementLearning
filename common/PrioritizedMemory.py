#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 14:14
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : PrioritizedMemory.py
# @Software: PyCharm
from common import Memory


class PrioritizedMemory(Memory):
    """
    采用SumTree实现的优先回放经验池
    """

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, alpha: float = 0.6):
        """
        构造函数
        :param obs_dim: 状态空间的维度
        :param size: 经验池的容量大小
        :param batch_size: 采样大小
        """

        super(PrioritizedMemory, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        tree_catcaity = 1
        while tree_catcaity < self.max_size:
            tree_catcaity *= 2
