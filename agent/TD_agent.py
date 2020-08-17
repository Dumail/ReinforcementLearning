# -*- coding:utf-8 -*-
# @Time : 2020/8/1 15:28
# @Author: PCF
# @File : TDAgent.py
from gym import Env
import random
from math import exp
from collections import defaultdict
from agent.Agent import Agent


class TDAgent(Agent):
    """
    使用TD算法的表格型智能体基类
    """

    def __init__(self, env: Env, random_q=False):
        """
        构造函数

        :param env: 智能体所在的环境，参照gym环境
        :param random_q: Q函数初值是随机还是全0
        """
        super().__init__(env)
        self.Q = {}  # 状态行为价值函数，使用双层dict实现，且键值均转换为对应的字符串
        # TODO 使用defaultdict实现Q函数
        self.random_q = random_q

    def reset(self):
        """
        初始化智能体状态，主要在多次训练中使用
        """
        super(TDAgent, self).reset()
        self.Q = {}

    def policy(self, qs, use_epsilon, episode=0):
        """
        根据贪心策略在行为空间中选取动作

        :param qs: 处于s状态，用于动作选取的状态动作函数q
        :param use_epsilon: true:采用epsilon贪心策略 false:采用贪心策略
        :param episode: 当前episode数，用于决定epsilon值
        :return: 选择的动作
        """
        epsilon = 1.0 / (episode + 1)
        if not use_epsilon or random.random() > epsilon:
            return max(qs, key=qs.get)
        else:
            return self.action_space.sample()

    def temperature(self, qs, t0=0.5, tk=0.05, episode=0):
        """
        模拟退火法选择动作，参见MaxPain算法

        :param qs: 处于s状态，用于动作选取的状态动作函数q
        :param t0: 初始温度
        :param tk: 温度衰减控制因子
        :param episode: 当前迭代次数，使温度随时间降低
        :return: 选择的动作
        """
        tau = t0 / (1 + tk * episode)
        temp_sum = 0
        for v in qs.values():
            temp_sum += exp(v / tau)
        acts, values, tmp = [0], [0], 0
        for k, v in qs.items():
            acts.append(k)
            values.append(exp(v / tau) / temp_sum + tmp)
            tmp = values[len(values) - 1]
        rand = random.random()
        for i in range(0, len(values)):
            if values[i + 1] >= rand >= values[i]:
                return acts[i + 1]

    def _is_in_q(self, q, s):
        """
        某状态下的状态动作价值函数是否初始化

        :param q: 状态动作价值函数
        :param s: 当前状态
        :return: true：已初始化 false:未初始化
        """
        return q.get(str(s)) is not None

    def _init_q(self, q, s):
        """
        初始化q值，有初始化为0和随机初始化两种方法

        :param q: 状态动作函数
        :param s: 状态
        """
        str_s = str(s)
        q[str_s] = {}
        for a in range(self.action_space.n):
            q[str_s][a] = random.random() / 10 if self.random_q is True else 0.0

    def _assert_q(self, q, s):
        """
        确保q中有s这个状态，且已初始化
        :param q: 状态价值函数
        :param s: 状态
        """
        if not self._is_in_q(q, s):
            self._init_q(q, s)  # 未初始化则直接初始化

    def _set_q(self, q, s, a, value):
        self._assert_q(q, s)
        q[str(s)][a] = value

    def _get_q(self, q, s, a):
        self._assert_q(q, s)
        return q[str(s)][a]
