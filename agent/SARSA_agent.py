# -*- coding:utf-8 -*-
# @Time : 2020/8/1 15:29
# @Author: PCF
# @File : SarsaAgent.py
from agent.TD_agent import *


class SarsaAgent(TDAgent):
    """
    采用同策略SARSA算法进行状态动作价值函数更新的表格型智能体
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
            a0 = self.policy(self.Q[str(s0)], use_epsilon=True, episode=cur_episode)  # 选择起点处的动作
            is_done = False  # 判断episode是否结束
            rewards_pre_episode = 0
            while not is_done:  # 执行到episode结束
                s1, r1, is_done, info = self.action(a0)  # 执行动作，获取结果
                rewards_pre_episode = round(r1 + rewards_pre_episode, 1)
                self._assert_q(self.Q, s1)
                a1 = self.policy(self.Q[str(s1)], use_epsilon=True, episode=cur_episode)  # epsilon贪心策略选择下一步要执行的动作
                old_q = self._get_q(self.Q, s0, a0)
                new_q = self._get_q(self.Q, s1, a1)
                # TD误差
                delta = r1 + gamma * new_q
                # 更新q值
                self._set_q(self.Q, s0, a0, old_q + alpha * (delta - old_q))
                s0, a0 = s1, a1
                self.obs = s0

                if cur_episode > render_episode:  # 渲染环境的步数，减少训练时间
                    self.env.render()

            print("episode {0} is {1:.1f} rewards.".format(cur_episode, rewards_pre_episode))
            self.rewards.append(rewards_pre_episode)
