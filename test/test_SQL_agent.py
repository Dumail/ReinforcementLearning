# -*- coding:utf-8 -*-
# @Time : 2020/8/1 16:52
# @Author: PCF
from unittest import TestCase
from environment.gridworld import *
from agent.SQL_agent import *
from utils.vis import *


# @File : test_SQL_agent.py
class TestSQLAgent(TestCase):
    def setUp(self):
        self.env = SimpleGridWorld()
        self.env.reset()
        self.max_episode = 500
        self.agent = SQLAgent(self.env, omega1=1, omega2=1, lambda1=1, lambda2=1)
        self.agent.reset()

    def tearDown(self):
        self.env.close()

    def test_learning(self):
        self.agent.learning(gamma=0.99, alpha=0.1, max_episode=self.max_episode, render_episode=400)

    def test_rewards(self):
        self.agent.learning(max_episode=self.max_episode, render_episode=self.max_episode)
        show(self.agent, self.max_episode)
