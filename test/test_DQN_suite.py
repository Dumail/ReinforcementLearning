#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 11:32
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_suite.py
# @Software: PyCharm
import unittest

from test import test_DDQN_agent, test_DQN_agent, test_PerQ_agent

if __name__ == '__main__':
    """
    集成测试
    """
    suite = unittest.TestSuite()
    suite.addTest(test_DQN_agent.TestDQNAgent('test_learning'))
    suite.addTest(test_DQN_agent.TestDQNAgent('test_test'))
    suite.addTest(test_DDQN_agent.TestDDQNAgent('test_learning'))
    suite.addTest(test_DDQN_agent.TestDDQNAgent('test_test'))
    suite.addTest(test_PerQ_agent.TestPerQAgent('test_learning'))
    suite.addTest(test_PerQ_agent.TestPerQAgent('test_test'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
