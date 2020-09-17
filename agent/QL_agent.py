# -*- coding:utf-8 -*-
# @Time : 2020/8/1 15:29
# @Author: PCF
# @File : QLAgent.py
from agent.TD_agent import *


class QLAgent(TDAgent):
    """
    采用异策略Q-Learning算法进行状态动作价值函数更新的表格型智能体
    """

    def learning(self, gamma=0.99, alpha=0.9, max_episode=500, render_episode=0):
        """
        SARSA学习算法
        :param gamma: 折扣
        :param alpha: 学习率
        :param max_episode: 最大训练episode次数
        :param render_episode: 开始渲染环境所需要的episode数，渲染会严重降低训练速度
        """
        for cur_episode in range(max_episode):
            s0 = self.obs = self.env.reset()
            self._assert_q(self.Q, s0)
            is_done = False
            reward_cur_episode = 0
            while not is_done:
                a0 = self.policy(self.Q[str(s0)], use_epsilon=True, episode=cur_episode)
                # a0 = self.temperature(self.Q[str(s0)], episode=cur_episode)
                s1, r1, is_done, info = self.action(a0)
                reward_cur_episode = round(r1 + reward_cur_episode, 1)
                self._assert_q(self.Q, s1)
                # 根据贪心策略选择目标动作
                ap = self.policy(self.Q[str(s1)], use_epsilon=False, episode=cur_episode)  # 采用贪心策略选择下一步的动作
                old_q = self._get_q(self.Q, s0, a0)
                new_q = self._get_q(self.Q, s1, ap)
                delta = r1 + gamma * new_q
                self._set_q(self.Q, s0, a0, old_q + alpha * (delta - old_q))  # 异策略更新
                self.obs = s0 = s1

                if cur_episode > render_episode:
                    self.env.render()  # 渲染环境

            print("episode {0} is {1:.1f} rewards.".format(cur_episode, reward_cur_episode))
            self.rewards.append(reward_cur_episode)
