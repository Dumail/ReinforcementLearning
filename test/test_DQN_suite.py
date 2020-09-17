#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 11:32
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_suite.py
# @Software: PyCharm
import unittest

from test import test_DDQN_agent, test_DQN_agent

if __name__ == '__main__':
    """
    集成测试
    """
    suite = unittest.TestSuite()
    suite.addTests([test_DQN_agent.TestDQNAgent('test_learning'),
                    test_DQN_agent.TestDQNAgent('test_test'),
                    test_DDQN_agent.TestDDQNAgent('test_learning'),
                    test_DDQN_agent.TestDDQNAgent('test_test')])
    runner = unittest.TextTestRunner()
    runner.run(suite)
