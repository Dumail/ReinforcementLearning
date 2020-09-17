#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 11:20
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_DDQN_agent.py
# @Software: PyCharm
import logging
import unittest

import gym

from agent.DDQN_agent import DDQNAgent


class TestDDQNAgent(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('DDQN test starting...')
        env = gym.make('CartPole-v0')
        self.path = '../weights/ddqn.pth'
        self.agent = DDQNAgent(env, memory_size=1000, batch_size=32, update_pred=100,
                               epsilon_decay=1 / 2000, path=self.path)

    def tearDown(self):
        logging.info('DDQN test ended!')

    def test_learning(self):
        self.agent.learning(max_episode=100)

    def test_test(self):
        self.agent.load_net(self.path)
        avg_score = self.agent.test(10)
        print("DDQN test average score is ", avg_score)
