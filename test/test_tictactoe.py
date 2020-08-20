# -*- coding:utf-8 -*-
# @Time : 2020/8/20 12:07
# @Author: PCF
from unittest import TestCase
from environment.tictactoe import TicTacToeEnv
from agent.QL_agent import QLAgent


# @File : test_tictactoe.py
class TestTicTacToeEnv(TestCase):

    def setUp(self):
        self.env = TicTacToeEnv(opponent_mode='random')
        self.env.reset()
        self.agent = QLAgent(self.env)
        self.agent.learning(max_episode=3000, render_episode=3000)

    def tearDown(self):
        self.env.close()
        # 统计测试的胜负结果
        print("O win {} times, X win {} times, draw {} times".format(self.result.count(1), self.result.count(-1),
                                                                     self.result.count(0)))

    def test_random_opponent(self):
        # 随机选择对战
        self.result = self.agent.test(times=50, render_time=50)

    def test_human_opponent(self):
        # 人机对战
        self.env = TicTacToeEnv(opponent_mode='human')
        self.agent.change_env(self.env)
        self.result = self.agent.test(times=5)
