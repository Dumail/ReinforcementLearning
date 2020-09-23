#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 15:34
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : Dueling_agent.py
# @Software: PyCharm
from agent.DQN_agent import DQNAgent
from common.networks import DuelingNetwork


class DuelingAgent(DQNAgent):
    def create_net(self):
        return DuelingNetwork(self.state_shape, self.action_shape).to(self.device)
