#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 9:42
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : DDQNAgent.py
# @Software: PyCharm
import torch

from agent.DQN_agent import DQNAgent


class DDQNAgent(DQNAgent):
    def compute_next_q_value(self, next_states: torch.FloatTensor):
        return self.target_net(next_states).gather(
            1, self.net(next_states).argmax(dim=1, keepdim=True)
        )
