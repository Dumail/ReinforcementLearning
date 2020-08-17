# -*- coding:utf-8 -*-
# @Time : 2020/8/16 17:36
# @Author: PCF
# @File : MP_agent.py
from agent.TD_agent import *


class MPAgent(TDAgent):
    """
    采用MaxPain算法更新状态动作价值函数的表格型智能体，参见MaxPain论文
    """

    def __init__(self, env: Env, omega, random_q=True):
        super(MPAgent, self).__init__(env, random_q)
        self.Qr = {}  # 正反馈对应的状态动作价值函数
        self.Qp = {}  # 痛苦对应的状态动作价值函数
        self.omega = omega

    def reset(self):
        super(MPAgent, self).reset()
        self.Qr = {}
        self.Qp = {}

    def _get_qs(self, s):
        qs1, qs2 = self.Qr[str(s)], self.Qp[str(s)]
        tmp = {}
        for key in qs1.keys():
            tmp[key] = self.omega * qs1.get(key) - (1 - self.omega) * qs2.get(key)
        return tmp

    def learning(self, gamma=0.99, alpha=0.9, max_episode=500, render_episode=0):
        for cur_episode in range(max_episode):
            s0 = self.obs = self.env.reset()
            self._assert_q(self.Qr, s0)
            self._assert_q(self.Qp, s0)
            is_done = False
            step = 0
            rewards_pre_episode = 0
            while not is_done:
                step += 1
                if cur_episode > render_episode:
                    self.env.render()
                # a0 = self.policy(self._get_qs(s0), use_epsilon=True, episode=cur_episode)
                a0 = self.temperature(self._get_qs(s0), episode=cur_episode)
                s1, r, is_done, info = self.action(a0)
                rewards_pre_episode = round(r + rewards_pre_episode, 1)
                self._assert_q(self.Qr, s1)
                self._assert_q(self.Qp, s1)
                # 结合两个q，贪心策略选择目标动作
                ap = self.policy(self._get_qs(s1), use_epsilon=False, episode=cur_episode)
                old_qr, old_qp = self._get_q(self.Qr, s0, a0), self._get_q(self.Qp, s0, a0)
                new_qr, new_qp = self._get_q(self.Qr, s1, ap), self._get_q(self.Qp, s1, ap)
                # 两个TD误差
                delta_r, delta_p = (r if r > 0 else 0) + gamma * new_qr, (
                    -r if r < 0 else 0) + gamma * new_qp
                # 分别更新两个q值
                self._set_q(self.Qr, s0, a0, old_qr + alpha * (delta_r - old_qr))
                self._set_q(self.Qp, s0, a0, old_qp + alpha * (delta_p - old_qp))
                self.obs = s0 = s1

            print("episode {0} is and {1} rewards.".format(cur_episode, rewards_pre_episode))
            self.rewards.append(rewards_pre_episode)
