# -*- coding:utf-8 -*-
# @Time : 2020/8/1 15:00
# @Author: PCF
# @File : main.py
from agent.QL_agent import QLAgent
from environment.gridworld import *

if __name__ == '__main__':
    env = SimpleGridWorld()
    env.reset()
    agent = QLAgent(env)
    agent.learning()
    # show(agent, 500)
    env.close()
