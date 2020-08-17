# -*- coding:utf-8 -*-
# @Time : 2020/8/17 21:02
# @Author: PCF
from unittest import TestCase
import gym
from agent.DQN_agent import DQNAgent
from environment.gridworld import *


# @File : test_DQN_agent.py
class TestDQNAgent(TestCase):
    def setUp(self):
        # self.env = gym.make("MountainCar-v0")
        self.env = SimpleGridWorld()
        self.env.reset()
        self.max_episode = 500
        self.agent = DQNAgent(self.env)
        self.agent.reset()

    def tearDown(self):
        self.env.close()

    def test_learning(self):
        self.agent.learning()
