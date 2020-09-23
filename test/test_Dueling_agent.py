#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 15:47
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_Dueling_agent.py
import gym
import logging
from unittest import TestCase

from agent.Dueling_agent import DuelingAgent


# @Software: PyCharm
class TestDuelingAgent(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('Dueling test starting...')
        env = gym.make('CartPole-v0')
        self.path = '../weights/dueling.pth'
        self.agent = DuelingAgent(env, memory_size=1000, batch_size=32, update_pred=100,
                                  epsilon_decay=1 / 2000, path=self.path)

    def tearDown(self):
        logging.info('Dueling test ended!')

    def test_learning(self):
        self.agent.learning(max_episode=300)

    def test_test(self):
        self.agent.load_net(self.path)
        avg_score = self.agent.test(10)
        logging.info("Dueling test average score is {}".format(avg_score))
