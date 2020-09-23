#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 11:32
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : test_suite.py
# @Software: PyCharm
import multiprocessing as mp
import unittest

from test import test_DDQN_agent, test_DQN_agent, test_PerQ_agent, test_Dueling_agent


def job(test1, test2):
    """
    单个处理器的的测试任务
    :param test1: 训练测试
    :param test2: 测试测试
    """
    suite = unittest.TestSuite()
    suite.addTest(test1)
    suite.addTest(test2)
    runner = unittest.TextTestRunner()
    runner.run(suite)


def multiprocessing_test():
    """
    多进程集成测试
    """
    p1 = mp.Process(target=job, args=(
        test_DQN_agent.TestDQNAgent('test_learning'),
        test_DQN_agent.TestDQNAgent('test_test')
    ))

    p2 = mp.Process(target=job, args=(
        test_DDQN_agent.TestDDQNAgent('test_learning'),
        test_DDQN_agent.TestDDQNAgent('test_test')
    ))

    p3 = mp.Process(target=job, args=(
        test_PerQ_agent.TestPerQAgent('test_learning'),
        test_PerQ_agent.TestPerQAgent('test_test')
    ))
    p4 = mp.Process(target=job, args=(
        test_Dueling_agent.TestDuelingAgent('test_learning'),
        test_Dueling_agent.TestDuelingAgent('test_test')
    ))
    p1.start()
    p2.start()
    p3.start()
    p4.start()


def normal_test():
    """
    常规集成测试，使用单进程或GPU
    :return:
    """
    suite = unittest.TestSuite()
    suite.addTest(test_DQN_agent.TestDQNAgent('test_learning'))
    suite.addTest(test_DQN_agent.TestDQNAgent('test_test'))
    suite.addTest(test_DDQN_agent.TestDDQNAgent('test_learning'))
    suite.addTest(test_DDQN_agent.TestDDQNAgent('test_test'))
    suite.addTest(test_PerQ_agent.TestPerQAgent('test_learning'))
    suite.addTest(test_PerQ_agent.TestPerQAgent('test_test'))
    suite.addTest(test_Dueling_agent.TestDuelingAgent('test_learning'))
    suite.addTest(test_Dueling_agent.TestDuelingAgent('test_test'))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    """
    集成测试
    """
    # normal_test()
    multiprocessing_test()
