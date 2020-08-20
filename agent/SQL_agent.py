# -*- coding:utf-8 -*-
# @Time : 2020/8/1 16:41
# @Author: PCF
# @File : SQL_agent.py
from agent.TD_agent import *


class SQLAgent(TDAgent):
    """
    采用Split Q-Learning算法进行状态动作价值函数更新的表格型智能体，参见SQL论文
    """

    def __init__(self, env: Env, omega1, omega2, lambda1, lambda2, random_q=True):
        """
        构造函数，参数设置见论文中表1
        """
        super(SQLAgent, self).__init__(env, random_q)
        self.Q1 = {}  # 正奖励对应的状态动作价值函数
        self.Q2 = {}  # 负奖励对应的状态动作价值函数
        self.omega1 = omega1
        self.omega2 = omega2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.random_q = random_q

    def reset(self):
        super(SQLAgent, self).reset()
        self.Q1 = {}
        self.Q2 = {}

    def _get_qs(self, s):
        """
        结合Q1,Q2，获得动作选取需要的总Q值
        :param s: 当前状态
        :return: q函数中当前状态对应的字典
        """
        qs1, qs2 = self.Q1[str(s)], self.Q2[str(s)]
        tmp = {}
        for key in qs1.keys():
            tmp[key] = qs1.get(key) + qs2.get(key)
        return tmp

    def learning(self, gamma=0.99, alpha=0.9, max_episode=500, render_episode=0):
        """
        SQL学习算法，基本过程于QL算法相同
        :param gamma: 折扣
        :param alpha: 学习率
        :param max_episode: 最大训练episode次数
        :param render_episode: 开始渲染环境所需要的episode数，渲染会严重降低训练速度
        """
        for cur_episode in range(max_episode):
            s0 = self.obs = self.env.reset()
            self._assert_q(self.Q1, s0)
            self._assert_q(self.Q2, s0)
            is_done = False
            rewards_pre_episode = 0
            while not is_done:
                a0 = self.policy(self._get_qs(s0), use_epsilon=True, episode=cur_episode)
                s1, r, is_done, info = self.action(a0)
                rewards_pre_episode = round(r + rewards_pre_episode, 1)
                self._assert_q(self.Q1, s1)
                self._assert_q(self.Q2, s1)
                # 在两个q中分别通过贪心策略选择目标动作
                ap1 = self.policy(self.Q1[str(s1)], use_epsilon=False, episode=cur_episode)
                ap2 = self.policy(self.Q2[str(s1)], use_epsilon=False, episode=cur_episode)
                old_q1, old_q2 = self._get_q(self.Q1, s0, a0), self._get_q(self.Q2, s0, a0)
                new_q1, new_q2 = self._get_q(self.Q1, s1, ap1), self._get_q(self.Q2, s1, ap2)
                # 正负奖励的TD误差,通过Omega调节即时奖励权重
                delta1, delta2 = self.omega1 * (r if r > 0 else 0) + gamma * new_q1, self.omega2 * (
                    r if r < 0 else 0) + gamma * new_q2
                # 分别更新两个Q值，通过lambda调节旧估计值权重
                self._set_q(self.Q1, s0, a0, self.lambda1 * old_q1 + alpha * (delta1 - old_q1))
                self._set_q(self.Q2, s0, a0, self.lambda2 * old_q2 + alpha * (delta2 - old_q2))
                self.obs = s0 = s1

                if cur_episode > render_episode:
                    self.env.render()

            print("episode {0} is and {1} rewards.".format(cur_episode, rewards_pre_episode))
            self.rewards.append(rewards_pre_episode)
