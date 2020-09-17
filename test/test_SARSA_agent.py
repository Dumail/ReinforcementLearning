# -*- coding:utf-8 -*-
# @Time : 2020/8/1 16:22
# @Author: PCF
from unittest import TestCase

from agent.SARSA_agent import *
from common.vis import *
from environment.gridworld import *


# @File : test_SARSA_agent.py
class TestSarsaAgent(TestCase):

    def setUp(self):
        self.env = SimpleGridWorld()
        self.env.reset()
        self.max_episode = 500
        self.agent = SarsaAgent(self.env)
        self.agent.reset()

    def tearDown(self):
        self.env.close()

    def test_learning(self):
        self.agent.learning(gamma=0.99, alpha=0.1, max_episode=self.max_episode, render_episode=0)

    def test_rewards(self):
        self.agent.learning(max_episode=self.max_episode, render_episode=self.max_episode)
        show(self.agent, self.max_episode)