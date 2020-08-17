# -*- coding:utf-8 -*-
# @Time : 2020/8/17 11:24
# @Author: PCF
# @File : Agent.py
from gym import Env


class Agent(object):
    """
    所有RL智能体基类
    """

    def __init__(self, env: Env):
        """
        构造函数
        :param env: 智能体所在的环境，参照gym环境
        """
        self.env = env
        self.action_space = env.action_space  # 智能体的行为空间由环境决定
        self.obs = None  # 智能体能观测到的自身状态
        self.rewards = []  # 记录每个episode的累计奖励

    def reset(self):
        """
        初始化智能体状态，主要在多次训练中使用
        """
        self.obs = None
        self.rewards = []

    def action(self, action):
        """
        在环境中执行行为

        :param action: 待执行的行为，应在行为空间中
        :return: 动作执行结果，参见gym接口
        """
        return self.env.step(action)

    def policy(self):
        """在行为空间中选取动作"""
        raise NotImplementedError

    def learning(self,gamma, alpha, max_episode, render_episode):
        """
        需要子类智能体具体实现的学习方法

        :param gamma: 折扣
        :param alpha: 学习率
        :param max_episode: 最大训练episode数
        :param render_episode: 在此episode后开始渲染环境
        """
        raise NotImplementedError
