# -*- coding:utf-8 -*-
# @Time : 2020/8/1 15:34
# @Author: PCF
from unittest import TestCase
from environment.gridworld import *


# @File : test_gridworld.py
class TestGrid(TestCase):
    def test_random_action(self):
        env = GridWorldEnv()
        env.reset()
        nfs = env.observation_space
        nfa = env.action_space
        print("nfs:%s; nfa:%s" % (nfs, nfa))
        print(env.observation_space)
        print(env.action_space)
        print(env.state)
        env.render()
        for _ in range(20000):
            env.render()
            a = env.action_space.sample()  # 在状态空间中随机选择动作
            state, reward, is_done, info = env.step(a)
            print("{0}, {1}, {2}, {3}".format(a, reward, is_done, info))
        print("env closed")
