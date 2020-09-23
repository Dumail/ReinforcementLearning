#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 15:41
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_PerQ_agent.py
import logging
from unittest import TestCase

# @Software: PyCharm
import gym

from agent.PerQ_agent import PerQAgent


class TestPerQAgent(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('DQN test starting...')
        env = gym.make('CartPole-v0')
        # env = MountainCarWrapper(gym.make('MountainCar-v0'))
        self.path = '../weights/per_dqn.pth'
        self.agent = PerQAgent(env, memory_size=1000, batch_size=32, update_pred=100,
                               epsilon_decay=1 / 2000, path=self.path)

    def tearDown(self):
        logging.info('Per DQN test ended!')

    def test_learning(self):
        self.agent.learning(max_episode=300)

    def test_test(self):
        self.agent.load_net(self.path)
        avg_score = self.agent.test(10)
        logging.info("Per DQN test average score is {}".format(avg_score))
