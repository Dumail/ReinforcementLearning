#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 10:01
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_DQN_agent.py
# @Software: PyCharm
import logging
import unittest

import gym

from agent.DQN_agent import DQNAgent


class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('DQN test starting...')
        env = gym.make('CartPole-v0')
        self.path = '../weights/dqn.pth'
        self.agent = DQNAgent(env, memory_size=1000, batch_size=32, update_pred=100,
                              epsilon_decay=1 / 2000, path=self.path)

    def tearDown(self):
        logging.info('DQN test ended!')

    def test_learning(self):
        self.agent.learning(max_episode=100)

    def test_test(self):
        self.agent.load_net(self.path)
        avg_score = self.agent.test(10)
        print("DQN test average score is ", avg_score)
